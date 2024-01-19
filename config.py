import os.path as osp
import argparse

class cfg():
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        # change
        self.data_root = osp.abspath(osp.join(self.this_dir,  '..'))


    def get_args(self):
        parser = argparse.ArgumentParser()
        # base

        parser.add_argument('--seq_len', default=512, type=int, help='The length of traing samples.')

        parser.add_argument('--gpu', default=0, type=int)
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--epoch', default=100, type=int)
        parser.add_argument('--embed_dim', default=1000, type=int, help='The embedding dimension in KGE model.')
        parser.add_argument('--max_len', type=int, default=100, help='max length of generated molecules')
        parser.add_argument('--min_len', type=int, default=100, help='min length of generated molecules')
        parser.add_argument('--num_words', type=int, default=70, help='min length of generated molecules')
        
        # ---------- generate ----------
        parser.add_argument('--return_num', type=int, default=250, help='number of return molecules')
        parser.add_argument('--beam', type=int, default=250, help='number of beams')
        parser.add_argument('--diversity_penalty', type=float, default=1.0, help='diversity penalty')
        parser.add_argument('--length_penalty', type=float, default=0.0, help='length penalty')
        
        parser.add_argument('--top_k', type=int, default=30, help='top k')
        parser.add_argument('--top_p', type=int, default=1, help='top p')
        parser.add_argument('--generate_mode', type=str, default='topk', help='generate mode: topk or beam search')
        parser.add_argument('--process', type=str, default='preprocess', help='generate selfies for finetune')
        parser.add_argument("--checkpoint_path", default='../moldata/dump/1129-EA_exp/BART_001/model/BartForConditionalGeneration_2000_epoch.pkl', type=str, help="model checkpoint path")
        parser.add_argument("--input_path", default='../moldata/finetune/plogp.csv', type=str, help="generate input data path")
        parser.add_argument("--output_path", default='../moldata/output/plogp.csv', type=str, help="generate output data path")
        parser.add_argument('--property', type=str, default='plogp', help='property to optimize')
        parser.add_argument('--penalty_alpha', type=float, default=0.0, help='The values balance the model confidence and the degeneration penalty in contrastive search decoding.')
        parser.add_argument('--temperature', type=float, default=1.0, help='The value used to module the next token probabilities.')
        
        parser.add_argument('--margin', default=9.0, type=float, help='The fixed margin in loss function. ')
        
        parser.add_argument('--adv_temp', default=1.0, type=float, help='The temperature of sampling in self-adversarial negative sampling.')
        parser.add_argument("--contrastive_loss", default=0, type=int, choices=[0, 1])
        parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        parser.add_argument('--normalize', type=bool, default=True, help='normalize predicited likelihood')
        parser.add_argument('--score_mode', type=str, default='log', help='use log-likelihood for ranking loss')
        parser.add_argument('--cand_margin', type=float, default=0.001, help='margin for ranking loss on candidate')
        parser.add_argument('--gold_margin', type=float, default=0, help='margin for ranking loss on gold')
        parser.add_argument('--gold_weight', type=float, default=0, help='weight for ranking loss on gold')
        parser.add_argument('--mle_weight', type=float, default=1, help='weight for mle loss on gold')
        parser.add_argument('--sim_weight', type=float, default=0, help='weight for sim loss on gold')
        parser.add_argument('--rank_weight', type=float, default=1, help='weight for ranking loss on candidate')
        parser.add_argument('--smooth', type=float, default=0.1, help='label smoothing')
        parser.add_argument('--valid_path', type=str, help="valid data path")
        parser.add_argument('--prefix_sequence_length', type=int, default=5, help="prefix sequence length")
        parser.add_argument('--mid_dim', type=int, default=512, help="middle dimensional for prefix")

        
        # ------------ 并行训练 ------------
        # 是否并行
        parser.add_argument('--rank', type=int, default=0, help='rank to dist')
        parser.add_argument('--dist', type=int, default=0, help='whether to dist')

        parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

        parser.add_argument('--world-size', default=3, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
        parser.add_argument("--local_rank", default=-1, type=int)
        

        self.cfg = parser.parse_args()
        return self.cfg
        

    def update_train_configs(self):
        # add some constraint for parameters
        # e.g. cannot save and test at the same time
        assert not (self.cfg.save_model and self.cfg.only_test)

        # TODO: update some dynamic variable
        self.cfg.data_root = self.data_root
        self.cfg.exp_id = f"{self.cfg.model_name}_{self.cfg.exp_id}"
        self.cfg.data_path = osp.join(self.data_root, self.cfg.data_path)
        self.cfg.dump_path = osp.join(self.cfg.data_path, self.cfg.dump_path)
        if self.cfg.only_test == 1:
            self.save_model = 0
            self.dist = 0
        
        return self.cfg
