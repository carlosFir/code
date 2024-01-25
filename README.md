# code

1.Update dataloader.py, add data preprocessing for generation.

2.Updata config-small.conf, add some parameters for generation.

3.Update main.py, split generate branch by "if... else..." controlled by parameters in config-small.conf.

4.Debug the data pipeline for generation of greedy search.

5.Decode the generated SMILES by using MyTokenizer().

6.Debug the data pipeline for generation of different beam_size.

7.Test the generation output by using the first half of SMILES to predict the other half.

8.(2024/1/25)Fix bug: When input length is 2, generated length can not be calculated correctly.

9.(2024/1/25)Add parameters in config for convenient saving when training. See 'save_model' in [train]

10.(2024/1/25)Add function 'CheckValid()' to check the decoded SMILES. But strings like 'xxx.xxx>>xxx!' can not be processed yet.

11.(2024/1/25)Print parameters of model and training so they can be recorded in the 'xxx.out' file when using 'nohup'

# to do 
1. Parameters in generation_configs are to be set reasonable.
2. More SMILES in file 'generate.txt' to test
3. Make the output correct. Now the generated SMILES are always made of too mant 'C's(CARBON TOKEN) and '(', ')' can't be matched grammatically correctly.

# Fix bug:

   ## CUDA problem with too long 'max_gen_len'
   When 'max_gen_len' is longer than 56, an error will be reported. It is something ralated to CUDA or index error.
   
   ## 'CheckValid()' is incompatible with reaction output
   When the decoded SMILES is like 'xxx.xxx', 'CheckValid()' can make correct judgement. When it is like 'xxx.xxx>>xxx!', 'CheckValid()' always prints 'False' because substring '>>' is not included in original tokens of SMILES. To fix this, 'CheckValid()' should be rewrited.

   ## End token '!' raises a shape problem because of the incorrect use of concatenation
   When bug CUDA problem is not reported (max_gen_len is no more than 56), a '!' token in generating process will raise an error of shape problem. Because '!' is the end token, the length of output will not be consisted with the expected length. Then 'torch.empty()' creates a shape-inconsistant tensor, resulting in error raised during concatenation. So the concat process should be recorrected.











