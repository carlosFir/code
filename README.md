# code

1.Update dataloader.py, add data preprocessing for generation.

2.Updata config-small.conf, add some parameters for generation.

3.Update main.py, split generate branch by "if... else..." controlled by parameters in config-small.conf.

4.Debug the data pipeline for generation of greedy search.

5.Decode the generated SMILES by using MyTokenizer().

6.Debug the data pipeline for generation of different beam_size.

7.Test the generation output by using the first half of SMILES to predict the other half.

# to do 
1. Parameters in generation_configs are to be set reasonable.
2. More SMILES in file
3. Make the output correct. Now the generated SMILES are always made of too mant 'C's(CARBON TOKEN) and '(', ')' can't be matched grammatically correctly.







