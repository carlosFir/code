# code
Update to generate on a file without decoding

1.Update dataloader.py, add data preprocessing for generation.

2.Updata config-small.conf, add some parameters for generation.

3.Update main.py, split generate branch by "if... else..." controlled by parameters in config-small.conf.

4.Debug the data pipeline for generation.

# to do 
1. Decode of the generated seqs.
2. Add end_token '!' to generated seqs.
3. Parameters in generation_configs are to be set reasonable.


Example:

Tensor 'seqs' after dataloader:

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

# 过程
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


