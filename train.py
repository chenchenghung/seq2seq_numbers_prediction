from dataset import train_data_loader
from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2seq
from torch.optim import Adam
import torch.nn.functional as F
import config
from tqdm import tqdm
import torch
from itertools import combinations
#訓練流程
#1.實例化model, optimizer, loss
#2.遍例dataloader
#3.調用得到output
#4.計算loss
#5.模型保存加載

seq2seq=Seq2seq().to(config.device)
optimizer=Adam(seq2seq.parameters(),lr=0.001)
def train(epoch):
    bar= tqdm(enumerate(train_data_loader),total=len(train_data_loader),ascii=True,desc='train')
    for idx,(input,target,input_length,target_length) in bar:
        input=input.to(config.device)
        target= target.to(config.device)
        input_length=input_length.to(config.device)
        target_length=target_length.to(config.device)
        optimizer.zero_grad()
        decoder_outputs,_=seq2seq(input,target,input_length,target_length)
        loss = F.nll_loss(decoder_outputs.view(-1,len(config.num_sequence)),target.view(-1),ignore_index=config.num_sequence.PAD)
        # decoder_outputs=decoder_outputs.view(decoder_outputs.size(0)*decoder_outputs.size(1),-1)#[batch_size*seq_len,-1]
        # target= target.view(-1) #[batch_size*seq_len][2,3,4]
        # loss= F.nll_loss(decoder_outputs,target,ignore_index=config.num_sequence.PAD)
        loss.backward()
        optimizer.step()
        bar.set_description('epoch:{}\tidx:{}\tloss:{:.2f}'.format(epoch,idx,loss.item()))
        if idx %100 ==0:
            torch.save(seq2seq.state_dict(),config.model_save_path)
            torch.save(optimizer.state_dict(),config.optimizer_save_path)



# encoder=Encoder()
# decoder=Decoder()
# print(encoder)
# print(decoder)
# for input, target, input_length, target_length in train_data_loader:
#     out,encoder_hidden=encoder(input,input_length)
#     decoder_outputs,decoder_hidden=decoder(target,encoder_hidden)
#     break

if __name__ == '__main__':
    for i in range(5):
        train(i)





