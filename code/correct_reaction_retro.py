from util.dataloader import *
from util.rdkit_tools import cal_props, cal_scaffold, get_mol
import tqdm
react_path = '/home/data/wd/norm_reaction.txt'
retro_path = '/home/data/wd/norm_retro_no_ca.txt'
def generate_reaction_data(txt_files):
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

def generate_retro_data(txt_files):
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

reactions = generate_reaction_data([react_path])
retros = generate_retro_data([retro_path])
print(len(reactions), len(retros))
valid_reactions = []
valid_retros = []
for i, react in enumerate(tqdm.tqdm(reactions)):
    try:
        props, _ = process_chem_item(react)
        if props is not None:
            valid_reactions.append(react[1])
    except:
        continue
for i, retro in enumerate(tqdm.tqdm(retros)):
    try:
        props, _ = process_chem_item(retro)
        if props is not None:
            valid_retros.append(retro[1])
    except:
        continue  
print(len(valid_reactions), len(valid_retros))

save_react_path = '/home/data/wd/correct_norm_reaction.txt'
save_retro_path = '/home/data/wd/correct_norm_retro_no_ca.txt'

print('saving...')
save_react_file = open(save_react_path, 'w')
for react in valid_reactions:
    save_react_file.write(react+'\n')
save_react_file.close()

save_retro_file = open(save_retro_path, 'w')
for retro in valid_retros:
    save_retro_file.write(retro+'\n')
save_retro_file.close()
print('saved')