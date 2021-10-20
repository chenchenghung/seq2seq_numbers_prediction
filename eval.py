'''
對模型進行評估
'''

from seq2seq import Seq2seq

import config
from tqdm import tqdm
import numpy as np
import torch
#訓練流程
#1.準備測試數據

data=[str(i) for i in np.random.randint(0,1e8,size=[100])] #input
data=sorted(data,key=lambda x : len(x),reverse=True)
input_length= torch.LongTensor([len(i) for i in data]).to(config.device)  #input_length
input= torch.LongTensor([config.num_sequence.transform(list(i),config.max_len) for i in data]).to(config.device)
# print('input_size:',input,input_length)
#2.實例化model, optimizer, loss
seq2seq=Seq2seq().to(config.device)
seq2seq.load_state_dict(torch.load(config.model_save_path))

#3.獲取預測值
indices= seq2seq.evaluate(input,input_length)
# print('indices:',indices)
indices=np.array(indices).transpose()
# print('indices:',indices)

#4.反序列化
result=[]
for line in indices:
    # print(line)
    temp_result=config.num_sequence.inverse_transform(line)
    cur_line=''
    for word in temp_result:
        if word == config.num_sequence.EOS_TAG:
            break
        cur_line += word
    result.append(cur_line)
print(data[:10])
print(result[:10])

