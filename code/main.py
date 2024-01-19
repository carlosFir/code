import torch
from util.Trainer import Trainer
from util.gpt2 import generate_model
from util.dataloader import ChemDataRaw, ChemGenerateDataRaw, MyTokenizer, get_data_loader, get_generate_data_loader, ChemDataSet
from util.TrainInits import init_seed, print_model_parameters
import configparser
from torch import nn 
import os
from util.gene_tools import DrawPlot


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
init_seed(114514)

# lr 0.0008 for first 1 epoch
# config
config = configparser.ConfigParser()
config.read("config-small.conf")
device = config['model']['device']

# tokenizer
vocab_file = config['data']['vocab_file']
tokenizer = MyTokenizer(vocab_file)
config['data']['vocab_size'] = str(tokenizer.get_vocab_size())

# model 
from_exist = bool(eval(config['train']['from_exist']))
if from_exist:
    exist_model = str(config['train']['exist_model'])
    print("Loading model from: ", os.path.join(config['train']['log_dir'], exist_model))
    model = torch.load(os.path.join(config['train']['log_dir'], exist_model))
else:
    model = generate_model(config)
model = model.to(device)
print_model_parameters(model, only_num=False)

# train or generate
generate = bool(eval(config['generate']['generation']))

if not generate:
    # dataset
    
    mol_files = config['data']['mol_files'].split(',')
    react_files = config['data']['react_files'].split(',')
    retro_files = config['data']['retro_files'].split(',')
    chemdata = ChemDataRaw(mol_files, react_files, retro_files)
    train_loader, val_loader, test_loader = get_data_loader(chemdata, 
                                                            int(config['train']['batch_size']), 
                                                            tokenizer,
                                                            int(config['model']['n_positions'])-1)
    if not from_exist:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            # else:
            #     nn.init.uniform_(p)
    optimizer = torch.optim.Adam(params=model.parameters(), 
                                lr=float(config['train']['lr_init']), 
                                eps=1.0e-8,
                                weight_decay=float(config['train']['weight_decay_rate']), 
                                amsgrad=False)
    
    trainer = Trainer(model, optimizer, train_loader, val_loader, test_loader, config)
   
    trainer.train()
else:
   
    model.eval()
    generation_configs = {
        'beam_size': 2,
        'max_gen_len': 5,
        'end_ids': 1, # 1能正常生成，但不知道效果如何；目前还没看generate中这个ids的作用
        'pad_ids': 1,
        'no_repeat_ngram_size': 0,
        'do_sample': True,
        'temperature': 1.0,
        'top_k': 10,
        'top_p': 0.8,
        'early_stop': True
    }
   
    generate_files = config['data']['generate_files'].split(',')
    scaffold = bool(eval(config['generate']['on_scaffold']))
    print("Generate on Scaffold: ", scaffold)

    # in dataloader.py, class ChemGenerateDataRaw is created for data preprocess in generation
    # ChemDataRaw中输出的example是'scaffold.smiles!'，返回的是smiles
    generate_data = ChemGenerateDataRaw(generate_files)
    
    # print(generate_data.generate_data)

    # in dataloader.py, def get_generate_data_loader is created for data loader in generation
    # 这里dataloader中ChemDataSetPostProcess引入了scaffold的表示
    # scaffold表示要考虑分子没有骨架的问题
    generate_loader = get_generate_data_loader(generate_data,
                                                scaffold,
                                                int(config['generate']['batch_size']),
                                                tokenizer,
                                                int(config['model']['n_positions'])-1)
    # print(generate_loader)
    
    for props, seqs, attn_mask in generate_loader:
        props = props.to(device)
        seqs = seqs.to(device)
        attn_mask = attn_mask.to(device)
        # print(seqs.shape, props.shape)
        print(seqs)
        # print(seqs[0])
        x = None
        for i in range(seqs.size(1)):
            # print(seqs[0, i])
            if seqs[0, i] != 68 and seqs[0, i+1] == 68:
                x = seqs[0, :i+1] # 删除end token
                break

        # Print the value of x
        print(x)
        if x != None:
            seqs = x.unsqueeze(0) # 删除end token
        
        print("seqs length: ", seqs.size(1))
        seqs_in = seqs[:, :(seqs.size(1) >> 1)]
        print(seqs_in.shape)
        len_in = seqs.size(1) >> 1
        print(len_in)
        half_generate = seqs.size(1) - len_in - 1# 待生成长度，取一半，奇数向上
        generation_configs['max_gen_len'] = half_generate # n means n + 1
        # out length = len_in + generation_configs['max_gen_len'] + 1
        generation_configs['beam_size'] = 2

        data = [props, seqs_in]
        
        print("#################Generating###############")
        out = model.generate(data, generation_configs=generation_configs)
        print(out.shape)
       
        
        out = out[:, :, 1:].squeeze(0) # 删去首位相接的重复token1个, remains = 20
        out_concat = torch.empty(generation_configs['beam_size'], len_in + generation_configs['max_gen_len'] + 1, dtype=torch.long)
        for i, seq in enumerate(out):
            s = torch.concat((seqs_in, seq.unsqueeze(0)), dim=1)
            out_concat[i] = s
            
        
        print("#####################Contrast####################")
        print('Seqs_all: ', seqs.shape, seqs)
        print('Generated: ', out_concat.shape, out_concat)
        '''print('Seqs_input: ', seqs_in.shape, seqs_in)
        print('Seqs_output: ', out.shape, out)'''

        print("################Decoding###############")
        for i, out in enumerate(out_concat):
            print(tokenizer.decode(out.tolist()))
        smiles_original = tokenizer.decode(seqs[0].tolist())
        print(smiles_original)
        # DrawPlot(out_concat, seqs)
        
        


   



'''
Total params num: 7549828
*****************Finish Parameter****************
2024-01-10 10:10: Experiment log path in: ./small_log
2024-01-10 10:10: Train Epoch 1: 0/62765 Loss: 22.319817
2024-01-10 11:50: Train Epoch 1: 10000/62765 Loss: 1.647095
2024-01-10 13:31: Train Epoch 1: 20000/62765 Loss: 1.050603
2024-01-10 15:11: Train Epoch 1: 30000/62765 Loss: 1.822520
2024-01-10 16:51: Train Epoch 1: 40000/62765 Loss: 1.250585
2024-01-10 18:32: Train Epoch 1: 50000/62765 Loss: 1.452319
2024-01-10 20:12: Train Epoch 1: 60000/62765 Loss: 1.354892
2024-01-10 20:40: **********Train Epoch 1: averaged Loss: 1.595912, tf_ratio: 0.000000
2024-01-10 20:42: **********Val Epoch 1: average Loss: 1.310222
2024-01-10 20:42: *********************************Current best model saved!
2024-01-10 20:42: Saving current best --- whole --- model to ./small_log/best_model.pth
'''
