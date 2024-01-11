import torch
from util.Trainer import Trainer
from util.gpt2 import generate_model
from util.dataloader import ChemDataRaw, ChemGenerateDataRaw, MyTokenizer, get_data_loader, get_generate_data_loader, ChemDataSet
from util.TrainInits import init_seed, print_model_parameters
import configparser
from torch import nn 
import os

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
    print("Loading model from: ", os.path.join(config['train']['log_dir'], 'best_model.pth'))
    model = torch.load(os.path.join(config['train']['log_dir'], 'best_model.pth'))
else:
    model = generate_model(config)
model = model.to(device)
print_model_parameters(model, only_num=False)

# train or generate
generate = bool(eval(config['train']['generate']))

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
        'beam_size': 1,
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
    # seqs:
    '''
    截断end token后：
    tensor([26, 49, 58, 39, 58, 23,  8, 23, 32, 23,  9, 23, 23, 23, 23, 23,  8,  9,
        59, 51, 23,  8, 23, 23, 23, 23, 23,  8, 35, 58, 58, 39, 58, 59, 39, 58,
        59, 23,  8, 23, 23, 23,  9, 32, 23, 23, 39, 58, 58, 39, 49, 26, 59, 51,
        23, 14, 23, 23, 23, 23, 23, 14,  7, 59, 23,  9, 23], device='cuda:6')
    截断end token前：
    tensor([[26, 49, 58,  ..., 68, 68, 68]], device='cuda:6')
    '''

    # end_ids=68, pad_ids=68 在n_positions=1024时，结果：
    '''
    tensor([[[23, 23, 58, 39, 39, 59, 58, 59, 23, 23, 23, 23, 23, 39, 58, 23, 23,
          23, 59, 23, 58, 39, 23, 58, 39, 39, 23, 59, 39, 23, 23, 23, 23, 23,
          39, 39, 58, 23, 23, 23, 58, 59, 23, 23, 39,  8, 58, 39, 59, 23, 23,
          58, 23, 23, 39, 58, 23, 23, 23,  8, 39,  8, 23, 23,  8, 39, 39, 59,
          23, 59, 23, 59, 39, 58]]], device='cuda:6')
    '''

    # end_ids=1, pad_ids=1 在n_positions=1024时，结果：
    '''
    tensor([[[23, 23, 58, 39, 39, 59, 58, 59, 23, 23, 23, 23, 23, 39, 58, 23, 23,
          23, 59, 23, 58, 39, 23, 58, 39, 39, 23, 59, 39, 23, 23, 23, 23, 23,
          39, 39, 58, 23, 23, 23, 58, 59, 23, 23, 39,  8, 58, 39, 59, 23, 23,
          58, 23, 23, 39, 58, 23, 23, 23,  8, 39,  8, 23, 23,  8, 39, 39, 59,
          23, 59, 23, 59, 39, 58]]], device='cuda:6')
    '''

    # ids=68 / 1 结果一样

    generate_files = config['data']['generate_files'].split(',')
    
    # in dataloader.py, class ChemGenerateDataRaw is created for data preprocess in generation
    generate_data = ChemGenerateDataRaw(generate_files)
    
    # print(generate_data.generate_data)

    # in dataloader.py, def get_generate_data_loader is created for data loader in generation
    generate_loader = get_generate_data_loader(generate_data,
                                                int(config['train']['batch_size']),
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
                x = seqs[0, :i] # 删除end token
                break

        # Print the value of x
        print(x)
        if x != None:
            seqs = x.unsqueeze(0) # 删除end token
        
        labels = [seqs, props]
        data = [props, seqs] # 已删除end token，直接一步到位取seqs的全部
        print("props: ", props.shape)
        print("seqs: ", seqs.shape)
        
        print("#################Generating###############")
        # （1）（已解决，问题（3）调教好了就可以直接放在cuda上了,n_positions测试到1024无报错）generate好像对n_positions很敏感，太大会报错
        # 在cuda上报错为RuntimeError: CUDA error: device-side assert triggered
        # 要在cpu上运行才能看到原始报错：IndexError: index out of range in self
        # 发现报错出现在对input_ids的传递中，print输出，发现了问题（3）
        # 问题（3）来源于token加入了end token
        # 问题（1）在gpt2.py的测试中，将seqs初始化为(2, 10) (2, 128)无报错
        # (2, 1024)报错相同
        # (2 ,512)正在测试，打印每一步结果

        # （2）（无所谓了，因为就算被截断了，查找不到end会使用原seqs）另外设置为较小值发现被截断的seq没有end token
        # 例如 n_positions=10
        # print(seqs[0]) = tensor([26, 49, 58, 39, 58, 23,  8, 23, 32, 23], device='cuda:6')
        # 截断的末尾应该也是!即68，不知道是否有影响，但1024够得够了，不会出现截断

        # （3）（已解决，在生成前规范化一下）另外，经过MyTokenizer的序列的末尾必定是68填充
        # generate中对初始input_ids也就是外部的data[1]即seqs的做法是取-1元素
        # 意味着input_ids一开始就是68的结尾符号，导致后面一直生成重复的68
        # 但为什么报错就不知道了
        # 需要重新设计token，不能加end token

        # 查看序列长度padding到n_positions是哪一步生成的
        # 查看一般小n_positions值的是否报错，报错是否相同
        # 查看极小n_positions值报错情况

        # （4）decoder
        # （5）生成的序列没有end token

       
        print(model.generate(data, generation_configs=generation_configs))

        break


   



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