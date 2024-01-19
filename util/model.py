import torch 
from torch import nn
import math
from attention import MemoryCompressedAttention as Attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0, lookup_index=None):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        # computing the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)  # (1, T_max, d_model)
        self.register_buffer('pe', pe)


    def forward(self, x):
        '''
        :param x: (batch_size, T, F_in)
        :return: (batch_size, T, F_out)
        '''
        if self.lookup_index is not None:
            x = x + self.pe[:, self.lookup_index, :]  # (batch_size, T, F_in) + (1,T,d_model)
        else:
            x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x.detach())

# PropEncoder 和 PropDecoder 的结构可以调整
class PropEncoder(nn.Module):
    def __init__(self, args):
        super(PropEncoder, self).__init__()
        self.prop_in_dim = args.prop_dim * args.prop_num
        self.embed_dim = args.embed_dim

        self.prop_encoder = nn.Sequential(
            nn.Linear(self.prop_in_dim, self.embed_dim),
            nn.GELU()
        )

    def forward(self, props):
        props = self.prop_encoder(props)
        return props


class PropDecoder(nn.Module):
    def __init__(self, args):
        super(PropDecoder, self).__init__()
        self.prop_dim = args.prop_dim
        self.prop_num = args.prop_num
        self.embed_dim = args.embed_dim

        self.prop_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.prop_in_dim),
        )

    def forward(self, props):
        return self.prop_decoder(props)


class GPTLayer(nn.Module):
    def __init__(self, d_model, seq_len, nhead=1, dim_feedforward: int = 64, dropout: float = 0.1):
        super(GPTLayer, self).__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, nhead, causal=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
    def forward(self, x):
        # x : B, L, d
        x = self.ln1(x)
        
        x = self.attn(x)
        
        x = x + self.dropout1(x)
        
        x = x + self.dropout2(self.ffn(self.ln2(x)))
        
        return x



class ChemGPT(nn.Module):
    def __init__(self, args):
        super(ChemGPT, self).__init__()
        
        self.max_len = args.max_len
        self.embed_dim = args.embed_dim
        self.num_words = args.num_words
        # modules
        self.embedding = nn.Embedding(args.num_words, self.embed_dim)
        self.PE = PositionalEncoding(self.embed_dim, self.max_len)
        self.prop_encoder = PropEncoder(args)
        self.prop_decoder = PropDecoder(args)
        self.attn_layers = nn.Sequential(*[GPTLayer(args.embed_dim, args.max_len, args.nhead) for _ in range(args.layers)])
        self.output_ln = nn.Linear(self.embed_dim, self.num_words)


    def forward(self, props, seq):
        # props : B, prop_num, prop_dim
        # seq : B, L, embed_dim
        props = self.prop_encoder(props) # B, 1, embed_dim

        # tokenize
        seq = self.tokenizer(seq)
        in_seq = torch.cat([props, seq], dim=1) # B, 1 + L, embed_dim
        out_seq = self.attn_layers(in_seq)

        # 最后一个词对应的特征用于预测性质
        props = out_seq[:, -1, :]
        props = self.prop_decoder(props)

        # 转回token
        out_seq = torch.softmax(self.output_ln(out_seq), dim=-1)

        pass

    def predict(self, props, seq):
        pass

    
