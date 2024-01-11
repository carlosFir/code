# code
Update to generate on a file without decoding

1.Update dataloader.py, add data preprocessing for generation.

2.Updata config-small.conf, add some parameters for generation.

3.Update main.py, split generate branch by "if... else..." controlled by parameters in config-small.conf.

4.Debug the data pipeline for generation.

Example:

Tensor 'seqs' after dataloader:

截断end token前：
tensor([[26, 49, 58,  ..., 68, 68, 68]], device='cuda:6')

截断end token后：

tensor([26, 49, 58, 39, 58, 23,  8, 23, 32, 23,  9, 23, 23, 23, 23, 23,  8,  9,
      59, 51, 23,  8, 23, 23, 23, 23, 23,  8, 35, 58, 58, 39, 58, 59, 39, 58,
      59, 23,  8, 23, 23, 23,  9, 32, 23, 23, 39, 58, 58, 39, 49, 26, 59, 51,
      23, 14, 23, 23, 23, 23, 23, 14,  7, 59, 23,  9, 23], device='cuda:6')
