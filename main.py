import torch
from util.Trainer import Trainer
from util.gpt2 import generate_model
from util.dataloader import ChemDataRaw, ChemGenerateDataRaw, MyTokenizer, get_data_loader, get_generate_data_loader, ChemDataSet
from util.TrainInits import init_seed, print_model_parameters
import configparser
from torch import nn 
import os
from util.gene_tools import DrawPlot, CheckValid



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
init_seed(114514)

# lr 0.0008 for first 1 epoch
# config
config = configparser.ConfigParser()
config.read("config-small.conf")
device = config['model']['device']
print(device)
# tokenizer
vocab_file = config['data']['vocab_file']
tokenizer = MyTokenizer(vocab_file)
config['data']['vocab_size'] = str(tokenizer.get_vocab_size())
# train or generate
generate = bool(eval(config['generate']['generation']))

# model 
from_exist = bool(eval(config['train']['from_exist']))

if from_exist:
    # exist_model = str(config['train']['exist_model'])
    if generate:
        exist_model = str(config['generate']['generate_model'])
    else:
        exist_model = str(config['train']['exist_model'])
    print("Loading model from: ", os.path.join(config['train']['log_dir'], exist_model))
    model = torch.load(os.path.join(config['train']['log_dir'], exist_model))
else:
    model = generate_model(config)

model = model.to(device)
# print_model_parameters(model, only_num=False)
total_num = sum([param.nelement() for param in model.parameters()])
print('Total params num: {}'.format(total_num))



if not generate:
    
    # parameters record
    print("--------------------model parameters: --------------------")
    print("n_positions: ", config['model']['n_positions'])
    print("n_embd: ", config['model']['n_embd'])
    print("n_layer: ", config['model']['n_layer'])
    print("n_head: ", config['model']['n_head'])
    print("device: ", config['model']['device'])

    print("--------------------train parameters: --------------------")
    print("learning_rate: ", config['train']['lr_init'])
    print("weight_decay_rate: ", config['train']['weight_decay_rate'])
    print("batch_size: ", config['train']['batch_size'])
    print("log_step: ", config['train']['log_step'])
    print("log_dir: ", config['train']['log_dir'])
    print("from_exist: ", config['train']['from_exist'])
    if config['train']['from_exist']:
        print("exist_model: ", config['train']['exist_model'])
    print("save_name: ", config['train']['save_name'])
          
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
    generate_files = config['data']['generate_files'].split(',')
    scaffold = bool(eval(config['generate']['on_scaffold']))
    print("Generate on Scaffold: ", scaffold)

    # in dataloader.py, class ChemGenerateDataRaw is created for data preprocess in generation
    # ChemDataRaw中输出的example是'scaffold.smiles!'，返回的是smiles
    generate_data = ChemGenerateDataRaw(generate_files)
    # in dataloader.py, def get_generate_data_loader is created for data loader in generation
    # 这里dataloader中ChemDataSetPostProcess引入了scaffold的表示
    # scaffold表示要考虑分子没有骨架的问题
    generate_loader = get_generate_data_loader(generate_data,
                                                scaffold,
                                                int(config['generate']['batch_size']),
                                                tokenizer,
                                                int(config['model']['n_positions'])-1)
    
    for props, seqs, attn_mask in generate_loader:
        props = props.to(device)
        seqs = seqs.to(device)
        attn_mask = attn_mask.to(device)

        x = None
        for i in range(seqs.size(1)):
            if seqs[0, i] != 68 and seqs[0, i+1] == 68:
                x = seqs[0, :i+1] # 删除end token
                break

        if x != None:
            seqs = x.unsqueeze(0) # 删除end token
        
        seqs_in = seqs[:, :(seqs.size(1) >> 1)]
        len_in = seqs.size(1) >> 1
        len_out = len_in
        generation_configs = {
            'beam_size': 1,
            'max_gen_len': 53,
            'end_ids': 68, # 1能正常生成，但不知道效果如何；目前还没看generate中这个ids的作用
            'pad_ids': 68,
            'no_repeat_ngram_size': 1,
            'do_sample': True,
            'temperature': 0.5,
            'top_k': 50,
            'top_p': 0.8,
            'early_stop': True
        }
        print(seqs_in.shape)
        print("#################Generating###############")
        data = [props, seqs_in]
        out = model.generate(data, generation_configs=generation_configs)
        out = out[:, :, 1:].squeeze(0) # 删去首位相接的重复token1个, remains = 20
        out_concat = torch.empty(generation_configs['beam_size'], len_in + generation_configs['max_gen_len'] + 1, dtype=torch.long)
        
        for i, seq in enumerate(out):
            s = torch.concat((seqs_in, seq.unsqueeze(0)), dim=1)
            out_concat[i] = s
     
        print("################Decoding###############")
        smiles_decoded = []
        for i, out in enumerate(out_concat):
            smiles_decoded.append(tokenizer.decode(out.tolist()))

        smiles_original = tokenizer.decode(seqs[0].tolist())
        print("Original SMILES: ")
        print(smiles_original)
        print("Decoded SMILES: ")
        for s in smiles_decoded:
            print(s, CheckValid(s))

        
        
        # RuntimeError: The expanded size of the tensor (58) must match the existing size (55) at non-singleton dimension 0.  Target sizes: [58].  Tensor sizes: [55]
        # to fix bug for long max_gen_len parameter
        # to fix bug for decoding output like 'xxx.xxx>>xxx!'
            
            
        
        


   



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
