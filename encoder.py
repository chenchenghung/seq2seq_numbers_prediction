#編譯器
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import config
import torch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.embedding= nn.Embedding(num_embeddings=len(config.num_sequence), #是字典總數 不是步長
                                     embedding_dim=config.embedding_dim,#用多長的向量去表示詞語
                                     padding_idx=config.num_sequence.PAD,
                                     )
        self.gru=nn.GRU(input_size=config.embedding_dim,
                        num_layers=config.num_layer,
                        hidden_size=config.hidden_size,
                        batch_first=True,
                        bidirectional=False,
                        dropout=0
                        )

    def forward(self,input,input_length):
        #param input: [batch size, max_len]
        embedded= self.embedding(input)  #embedded--->[batch_size,max_len,embedding_dim]
        embedded=pack_padded_sequence(embedded, input_length,batch_first=True) #打包

        out,hidden= self.gru(embedded)
        #解包
        out,out_length=pad_packed_sequence(out,batch_first=True,padding_value=config.num_sequence.PAD)
        #out: [batch_size,sen_len, hidden_size]
        #hidden: [1*1, batch_size, hidden_size ]
        return out,hidden


if __name__ == '__main__':


    from dataset import train_data_loader
    encoder=Encoder()
    print(encoder)
    for input, target, input_length, target_length in train_data_loader:
        out,hidden=encoder(input,input_length)
        print(out.size())
        print(hidden.size())
        # print(out_length)

        break
