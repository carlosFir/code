import transformers
import torch


if __name__=='__main__':
   
    
    print("here")
    props = torch.randn([2, 4]).to('cuda')
    print("here")
    seqs = torch.ones([2, 5]).long().to('cuda')



    
    