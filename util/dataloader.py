import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import re
from .rdkit_tools import cal_props, cal_scaffold, get_mol
import json
import tqdm
import random

'''
pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)
smiles += str('<')*(self.max_len - len(regex.findall(smiles)))
'''

def process_zinc(path, save_file):
    '''
    将ZINC数据集中的SMILES保存到save_file文件中
    '''
    all_smiles_list = open(save_file, '+w')
    for data_file in os.listdir(path):
        data_file = os.path.join(path, data_file)
        data = pd.read_csv(data_file, sep='\t')
        smiles_list = data['smiles'].tolist()
        for smiles in smiles_list:
            all_smiles_list.write(smiles+'\n')
    all_smiles_list.close()

def process_ustpo(path, save_file):
    '''
    *.rsmi文件 ReactionSmiles	PatentNumber	ParagraphNum	Year	TextMinedYield	CalculatedYield
    '''
    reaction_smiles_file = open(save_file, '+w')
    for data_file in os.listdir(path):
        if data_file[-4:]!='rsmi':
            continue
        data = pd.read_csv(os.path.join(path, data_file), sep='\t')
        reaction_smiles = data['ReactionSmiles']
        reaction_smiles_list = reaction_smiles.tolist()
        for smiles in reaction_smiles_list:
            reaction_smiles_file.write(smiles.split(' ')[0]+'\n')
    reaction_smiles_file.close()

def process_rxn(path, save_file):
    '''
    rid	tid	reactants	products	cats	sols
    '''
    data = pd.read_csv(path, sep='\t')
    def map(string):
        string = str(string).split('"')
        mol_list = []
        for item in string:
            if item in "[], ":
                continue
            mol_list.append(item)
        return mol_list
    reaction_smiles_file = open(save_file, '+w')
    reactants = data['reactants'].apply(map).tolist()
    products = data['products'].apply(map).tolist()
    cats = data['cats'].apply(map).tolist()
    sols = data['cats'].apply(map).tolist() 
    for i in range(len(reactants)):
        item = ''
        for rec in reactants[i]:
            item += rec + '.'
        if item[-1]!='.':
            continue
        item = item[:-1]+'>'
        for cat in cats[i]:
            item += cat + '.'
        for sol in sols[i]:
            item += sol + '.'
        if item[-1]=='.':
            item = item[:-1]+'>'
        else:
            item += '>'
        for prod in products[i]:
            item += prod + '.'
        if item[-1]!='.':
            continue
        item = item[:-1]
        reaction_smiles_file.write(item+'\n')
    reaction_smiles_file.close()

def process_reaction_smiles(smiles):
    '''
    smiles:  reactant.reactant>reagent>product.product
    '''
    temp = smiles.split('>')
    if len(temp)==3:
        reactants, reagents, products = temp[0], temp[1], temp[2]
    else:
        # print(smiles)
        return None
    try:
        props = cal_props(get_mol(products.split('.')[-1]))
    except:
        props = None
        # print(smiles)
    return {
        'type':'reaction',
        'reactants':reactants.split('.'),
        'reagents':reagents.split('.'),
        'products':products.split('.'),
        'props':props
    }

def process_retro_smiles(smiles):
    '''
    smiles:  reactant.reactant>reagent>product.product
    '''
    temp = smiles.split('<')
    if len(temp)==3:
        reactants, reagents, products = temp[0], temp[1], temp[2]
    else:
        # print(smiles)
        return None
    props = cal_props(get_mol(products.split('.')[-1]))
    return {
        'type':'reaction',
        'reactants':reactants.split('.'),
        'reagents':reagents.split('.'),
        'products':products.split('.'),
        'props':props
    }

def process_smiles(smiles):
    '''
    将SMILES组织称目标数据格式   props + scaffold + smiles
    '''
    mol = get_mol(smiles)
    props = cal_props(mol)
    scaffold = cal_scaffold(mol)
    return {
        'type':'molecular',
        'props':props,
        'scaffold':scaffold,
        'smiles':smiles
    }

def generate_reaction_json(txt_files, save_file):
    save_file = open(save_file, '+w')
    data = []
    for txt_file in txt_files:
        print('processing '+txt_file)
        txt_data = open(txt_file, 'r')
        for i, line in enumerate(tqdm.tqdm(txt_data)):
            if line[-1]=='\n':
                line = line[:-1]
                item = process_reaction_smiles(line)
                if item is not None:
                    data.append(item)
        txt_data.close()

    text = json.dumps(data)
    save_file.write(text)
    save_file.close()
    

def generate_smiles_json(txt_files, save_file, summary):
    # save_file = open(save_file, '+w')
    file_list = []
    data = []
    total = 0
    for txt_file in txt_files:
        txt_data = open(txt_file, 'r')
        for i, line in enumerate(tqdm.tqdm(txt_data)):
            if line[-1]=='\n':
                line = line[:-1]
            data.append(process_smiles(line))
            if len(data)==10000:
                text = json.dumps(data)
                save_file_i = save_file+'_'+str(i)+'_'+str(i+len(data))+'.json'
                file_list.append(save_file_i)
                save_file_i = open(save_file_i, '+w')
                save_file_i.write(text)
                data = []
            total+=10000
        if len(data)>0:
            text = json.dumps(data)
            save_file_i = save_file+'_'+str(total)+'_'+str(total+len(data))+'.json'
            file_list.append(save_file_i)
            save_file_i = open(save_file_i, '+w')
            save_file_i.write(text)
            data = []
        txt_data.close()

    s = open(summary, '+w')
    for file in file_list:
        s.write(file+'\n')
    s.close()

    # text = json.dumps(data)
    # save_file.write(text)
    # save_file.close() 


class ChemData():
    '''
    ChemData负责将raw txt style数据集转化成 json数组形式的格式化数据用于模型训练。
    json attributes:
    {
        'type':'reaction',
        'reactants':[reactants],
        'reagents':[reagents],
        'products':[products],
        'props':[props],
        'scaffold':scaffold,
        'smiles':smiles
    }
    '''
    def __init__(self, mol_data, reaction_data, ratios=[0.1,1,1], train_val_test_split=[0.9, 0.91, 1], tasks=['mol_gen','reaction','retro','prop']):
        self.data = []

        # 读原始数据 list of dicts
        molecules, reactions, retro = [], [], []
        if len(mol_data)>0:
            molecules = self.generate_mol_data(mol_data)
        if len(reaction_data)>0:
            reactions, retro = self.generate_reaction_retro_data(reaction_data)
        
        # 调整比例----
        random.shuffle(molecules)
        random.shuffle(reactions)
        random.shuffle(retro)
        # print(len(molecules),len(reactions),len(retro))
        molecules = molecules[:int(ratios[0]*len(molecules))]
        reactions = reactions[:int(ratios[1]*len(reactions))]
        retro = reactions[:int(ratios[2]*len(retro))]
        # print(len(molecules),len(reactions),len(retro))
        # -----------

        # 合并
        self.train_data = molecules[:int(train_val_test_split[0]*len(molecules))] + \
                            reactions[:int(train_val_test_split[0]*len(reactions))] + \
                            retro[:int(train_val_test_split[0]*len(retro))]
        
        self.val_data = molecules[int(train_val_test_split[0]*len(molecules)):int(train_val_test_split[1]*len(molecules))] + \
                            reactions[int(train_val_test_split[0]*len(reactions)):int(train_val_test_split[1]*len(reactions))] + \
                            retro[int(train_val_test_split[0]*len(retro)):int(train_val_test_split[1]*len(retro))]
        
        self.test_data = molecules[int(train_val_test_split[1]*len(molecules)):] + \
                            reactions[int(train_val_test_split[1]*len(reactions)):] + \
                            retro[int(train_val_test_split[1]*len(retro)):]



    def generate_mol_data(self, txt_files):
        molecules = []
        for txt_file in txt_files:
            if len(txt_file)==0:
                continue
            txt_data = open(txt_file, 'r')
            for i, line in enumerate(tqdm.tqdm(txt_data)):
                # if i==1000:
                #     break
                if line[-1]=='\n':
                    line = line[:-1]
                molecules.append(process_smiles(line)) # {'type':'molecular','smiles':line}
        return molecules

    def generate_reaction_retro_data(self, txt_files):
        reactions = []
        retro = []
        for txt_file in txt_files:
            print('processing '+txt_file)
            txt_data = open(txt_file, 'r')
            for i, line in enumerate(tqdm.tqdm(txt_data)):
                # if i==1000:
                #     break
                if line[-1]=='\n':
                    line = line[:-1]
                    item = process_reaction_smiles(line)
                    if item is not None and len(item['products'])==1:
                        # print('adding sample')
                        reactions.append(item)
                        item['type'] = 'retro'
                        try:
                            props = cal_props(get_mol(item['reactants'][-1]))
                        except:
                            props = None
                        item['props'] = props
                        retro.append(item)
        return reactions, retro
    
class ChemDataSet(Dataset):
    def __init__(self, data, tokenizer=None, max_length=500):
        self.data = data
        self.max_length = max_length
        if tokenizer is None:
            self.tokenizer = MyTokenizer()
        else:
            self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = self.data[index]
        # preprocess 将dict组织成字符串
        task = item['type']
        if task=='reaction':
            props = item['props']
            train_str = ''
            for reactant in item['reactants']:
                train_str += reactant+'.'
            if train_str[-1]=='.':
                train_str = train_str[:-1]+'>'
            for reactant in item['reagents']:
                train_str += reactant+'.'
            if train_str[-1]=='.':
                train_str = train_str[:-1]+'>'
            for reactant in item['products']:
                train_str += reactant+'.'
            if train_str[-1]=='.':
                train_str = train_str[:-1]+'!'
            # return props, train_str
        elif task=='retro':
            props = item['props']
            train_str = ''
            for reactant in item['products']:
                train_str += reactant+'.'
            if train_str[-1]=='.':
                train_str = train_str[:-1]+'<'
            for reactant in item['reagents']:
                train_str += reactant+'.'
            if train_str[-1]=='.':
                train_str = train_str[:-1]+'<'
            for reactant in item['reactants']:
                train_str += reactant+'.'
            if train_str[-1]=='.':
                train_str = train_str[:-1]+'!'
            # return props, train_str
        elif task=='molecular':
            item = process_smiles(item['smiles'])
            props = item['props']
            train_str = item['scaffold']+'.'+item['smiles']+'!'
            # return props, train_str
        else:
            raise KeyError('Wrong task!!!')
        train_str, attention_mask = self.tokenizer.encode(train_str, self.max_length)
        return torch.Tensor(props), train_str, attention_mask
    
    def __len__(self):
        return len(self.data)






# created for generation
def get_generate_data_loader(chemdata, on_scaffold, batch_size, tokenizer, max_length=500):
    return DataLoader(ChemGeneDataSetPostProcess(chemdata.generate_data, on_scaffold, tokenizer, max_length), batch_size, num_workers=8, shuffle=False)

def get_data_loader(chemdata, batch_size, tokenizer, max_length=500):
    return (DataLoader(ChemDataSetPostProcess(chemdata.train_data, tokenizer, max_length), batch_size, num_workers=8, shuffle=True), 
            DataLoader(ChemDataSetPostProcess(chemdata.val_data, tokenizer, max_length), batch_size, num_workers=8, shuffle=True), 
            DataLoader(ChemDataSetPostProcess(chemdata.test_data, tokenizer, max_length), batch_size, num_workers=8, shuffle=True))

class MyTokenizer():
    def __init__(self, charset_file='/home/data/wd/all_chars.txt'):
        self.char2num = {}
        cnt = 0
        charset_file = open(charset_file, 'r')
        for char in charset_file.readlines():
            char = char[:-1] if char[-1]=='\n' else char
            # print(char)
            self.char2num[char] = cnt
            cnt += 1
        self.num2char = {}
        for char in self.char2num.keys():
            self.num2char[self.char2num[char]] = char

    def get_vocab_size(self):
        return len(self.char2num)
    
    def get_eos_token(self):
        return '!'
    
    def encode(self, string, length=400):
        string = list(map(lambda x:self.char2num[x], string))
        string = string[:length] # truncate
        eos_token = self.get_eos_token()
        padding = [self.char2num[eos_token]]*(length-len(string))
        attention_mask = torch.zeros(length)
        # attention_mask需要考虑props位置，因此长度比string大1
        attention_mask[:len(string)+1] = 1      
        string = string + padding
        # print(string, padding)
        return torch.LongTensor(string), attention_mask


    def decode(self, index):
        return ''.join(list(map(lambda x:self.num2char[x], index)))

class ChemDataRaw():

    def __init__(self, mol_data, reaction_data, retro_data, ratios=[0.1,1,1], train_val_test_split=[0.9, 0.91, 1], tasks=['mol_gen','reaction','retro','prop']):
        self.data = []
        # 读原始数据 list of dicts
        molecules, reactions, retro = [], [], []
        if len(mol_data)>0:
            molecules = self.generate_mol_data(mol_data)
        if len(reaction_data)>0:
            reactions = self.generate_reaction_data(reaction_data)
        if len(retro_data)>0:
            retro = self.generate_retro_data(retro_data)
        # 调整比例
        random.shuffle(molecules)
        random.shuffle(reactions)
        random.shuffle(retro)

        print(len(molecules),len(reactions),len(retro))
        molecules = molecules[:int(ratios[0]*len(molecules))]
        reactions = reactions[:int(ratios[1]*len(reactions))]
        retro = retro[:int(ratios[2]*len(retro))]
        print(len(molecules),len(reactions),len(retro))
        # 合并
        self.train_data = molecules[:int(train_val_test_split[0]*len(molecules))] + \
                            reactions[:int(train_val_test_split[0]*len(reactions))] + \
                            retro[:int(train_val_test_split[0]*len(retro))]
        
        self.val_data = molecules[int(train_val_test_split[0]*len(molecules)):int(train_val_test_split[1]*len(molecules))] + \
                            reactions[int(train_val_test_split[0]*len(reactions)):int(train_val_test_split[1]*len(reactions))] + \
                            retro[int(train_val_test_split[0]*len(retro)):int(train_val_test_split[1]*len(retro))]
        
        self.test_data = molecules[int(train_val_test_split[1]*len(molecules)):] + \
                            reactions[int(train_val_test_split[1]*len(reactions)):] + \
                            retro[int(train_val_test_split[1]*len(retro)):]
        
        print('-------------- showing examples --------------------------------')
        print('molecules:')
        self.show_case(molecules[0])
        print('reactions:')
        self.show_case(reactions[0])
        print('retro:')
        self.show_case(retro[0])
        print('----------------------------------------------------------------')

    def show_case(self, case):
        props, train_str = process_chem_item(case)
        print(props, train_str)

    def generate_mol_data(self, txt_files):
        molecules = []
        for txt_file in txt_files:
            if len(txt_file)==0:
                continue
            txt_data = open(txt_file, 'r')
            for i, line in enumerate(tqdm.tqdm(txt_data)):
                if line[-1]=='\n':
                    line = line[:-1]
                # debug
                if i>100000:
                    break
                molecules.append(['mol', line]) 
            txt_data.close()
        return molecules

    def generate_reaction_data(self, txt_files):
        reactions = []
        for txt_file in txt_files:
            print('processing '+txt_file)
            txt_data = open(txt_file, 'r')
            for i, line in enumerate(tqdm.tqdm(txt_data)):
                if line[-1]=='\n':
                    line = line[:-1]
                    reactions.append(['rea', line])
            txt_data.close()
        return reactions
    
    def generate_retro_data(self, txt_files):
        retro = []
        for txt_file in txt_files:
            print('processing '+txt_file)
            txt_data = open(txt_file, 'r')
            for i, line in enumerate(tqdm.tqdm(txt_data)):
                if line[-1]=='\n':
                    line = line[:-1]
                    retro.append(['ret', line])
            txt_data.close()
        return retro
    
class ChemGenerateDataRaw():
    # no shuffle
    # no reaction
    # no retro
    # no split and ratios
    def __init__(self, mol_data, tasks=['mol_gen','prop']):
        self.data = []
        # 读原始数据 list of dicts
        molecules = []
        if len(mol_data)>0:
            molecules = self.generate_mol_data(mol_data)  
            # print("molecules: ", molecules)
        print("molecules length: ", len(molecules))

        self.generate_data = molecules
        '''# 合并
        self.train_data = molecules[:int(train_val_test_split[0]*len(molecules))] + \
                            reactions[:int(train_val_test_split[0]*len(reactions))] + \
                            retro[:int(train_val_test_split[0]*len(retro))]
        
        self.val_data = molecules[int(train_val_test_split[0]*len(molecules)):int(train_val_test_split[1]*len(molecules))] + \
                            reactions[int(train_val_test_split[0]*len(reactions)):int(train_val_test_split[1]*len(reactions))] + \
                            retro[int(train_val_test_split[0]*len(retro)):int(train_val_test_split[1]*len(retro))]
        
        self.test_data = molecules[int(train_val_test_split[1]*len(molecules)):] + \
                            reactions[int(train_val_test_split[1]*len(reactions)):] + \
                            retro[int(train_val_test_split[1]*len(retro)):]'''
        
        print('-------------- showing examples --------------------------------')
        print('molecules:')
        self.show_case(molecules[0])

    def show_case(self, case):
        props, train_str = process_chem_item(case)
        # print('train_sttr: ', train_str)
        print(props, train_str)

    def generate_mol_data(self, txt_files):
        molecules = []
        for txt_file in txt_files:
            if len(txt_file)==0:
                continue
            txt_data = open(txt_file, 'r')
            for i, line in enumerate(tqdm.tqdm(txt_data)):
                if line[-1]=='\n':
                    line = line[:-1]
                # debug
                if i>100000:
                    break
                molecules.append(['mol', line]) 
                
            txt_data.close()
        return molecules


class ChemDataSetPostProcess(Dataset):
    def __init__(self, data, tokenizer=None, max_length=500):
        self.data = data
        self.max_length = max_length
        if tokenizer is None:
            self.tokenizer = MyTokenizer()
        else:
            self.tokenizer = tokenizer

    def __getitem__(self, index):
        props, train_str = process_chem_item(self.data[index])
        train_str, attention_mask = self.tokenizer.encode(train_str, self.max_length+1)
        # print(props)
        # if props is None:
        #     exit()
        return torch.Tensor(props), train_str, attention_mask
    
    def __len__(self):
        return len(self.data)

class ChemGeneDataSetPostProcess(Dataset):
    def __init__(self, data, on_scaffold, tokenizer=None, max_length=500):
        self.data = data
        self.on_scaffold = on_scaffold
        self.max_length = max_length
        if tokenizer is None:
            self.tokenizer = MyTokenizer()
        else:
            self.tokenizer = tokenizer

    def __getitem__(self, index):
        props, generate_str = process_chem_item(self.data[index])
        seq_parts = generate_str.split('.')
        if not self.on_scaffold:
            generate_str = seq_parts[1] # 这里选择的是smiles做测试
        else: 
            generate_str = seq_parts[0] # 选择scaffold做测试
        # print("generate_str: ", generate_str)
        generate_str, attention_mask = self.tokenizer.encode(generate_str, self.max_length+1)
        # print(props)
        # if props is None:
        #     exit()
        return torch.Tensor(props), generate_str, attention_mask
    
    def __len__(self):
        return len(self.data)

def process_chem_item(case):
    task = case[0]
    item = case[1]
    if task=='rea':
        item = process_reaction_smiles(item)
        props = item['props']
        train_str = ''
        for reactant in item['reactants']:
            train_str += reactant+'.'
        if train_str[-1]=='.':
            train_str = train_str[:-1]+'>'
        for reactant in item['reagents']:
            train_str += reactant+'.'
        if train_str[-1]=='.':
            train_str = train_str[:-1]+'>'
        for reactant in item['products']:
            train_str += reactant+'.'
        if train_str[-1]=='.':
            train_str = train_str[:-1]+'!'
        # return props, train_str
    elif task=='ret':
        item = process_retro_smiles(item)
        props = cal_props(get_mol(item['reactants'][-1]))
        train_str = ''
        for reactant in item['products']:
            train_str += reactant+'.'
        if train_str[-1]=='.':
            train_str = train_str[:-1]+'<'
        for reactant in item['reagents']:
            train_str += reactant+'.'
        if train_str[-1]=='.':
            train_str = train_str[:-1]+'<'
        for reactant in item['reactants']:
            train_str += reactant+'.'
        if train_str[-1]=='.':
            train_str = train_str[:-1]+'!'
        # return props, train_str
    elif task=='mol':
        item = process_smiles(item)
        props = item['props']
        train_str = item['scaffold']+'.'+item['smiles']+'!'
        # return props, train_str
    else:
        raise KeyError('Wrong task!!!')
    return props, train_str
    



if __name__ == '__main__':
    print('in main')
    zinc_data_path = '/home/data/wd/zinc_data.txt'
    uspto_data_path = '/home/data/wd/USPTO/uspto_data.txt'
    rxn_data_path = '/home/data/wd/rxn.txt'
    all_data = ChemData([zinc_data_path], [uspto_data_path, rxn_data_path])
    dataset = ChemDataSet(all_data.val_data)
    print(dataset[0], len(dataset))
