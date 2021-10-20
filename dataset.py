#準備數據及
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import config


class NumDataset(Dataset):
    def __init__(self):
        #使用numpy隨機創建一堆數字
        np.random.seed(10)
        self.data=np.random.randint(0,1e8,size=[500000])


    def __getitem__(self, index):

        input=list(str(self.data[index]))
        target=input+['0']
        input_length= len(input)
        target_length= len(target)
        return input,target,input_length,target_length
    def __len__(self):
        return self.data.shape[0]

def collate_fn(batch):
    #param_batch: [(input,label,input_length,label_length),(input,label,input_length,label_length)]
    batch=sorted(batch, key=lambda x : x[3],reverse=True) #降序排序
    input,target,input_length,target_length=list(zip(*batch))
    #把input 轉為序列
    input= torch.LongTensor([config.num_sequence.transform(i,max_len=config.max_len) for i in input])
    target= torch.LongTensor([config.num_sequence.transform(i,max_len=config.max_len+1,add_eos=True) for i in target])
    input_length=torch.LongTensor(input_length)
    target_length=torch.LongTensor(target_length)
    return input,target,input_length,target_length

train_data_loader=DataLoader(NumDataset(),batch_size=config.train_batch_size,shuffle=True,collate_fn=collate_fn)

if __name__=='__main__':
    # num_dataset=NumDataset()
    # print(num_dataset.data[:10])
    # print(num_dataset[0])
    # print(len(num_dataset))
    for input,target,input_length,target_length in train_data_loader:
        print('input:',input[:3])
        print(input.size())
        print(target.size())
        print(target[:3])
        print('*'*10)
        print(input_length)
        print('*'*10)
        print(target_length)
        break
