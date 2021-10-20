#解碼器

import torch.nn as nn
import config
import torch
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.embedding=nn.Embedding(num_embeddings=len(config.num_sequence),
                                    embedding_dim=config.embedding_dim,
                                    padding_idx=config.num_sequence.PAD
                                    )
        self.gru=nn.GRU(input_size=config.embedding_dim,
                        hidden_size=config.hidden_size,
                        num_layers=config.num_layer,
                        batch_first=True)

        self.fc=nn.Linear(config.hidden_size,len(config.num_sequence))


    def forward(self,target,encoder_hidden):
        #1.獲取encoder的輸出作為decoder第一次的隱藏狀態
        decoder_hidden= encoder_hidden
        # batch_size=target.size(0)
        batch_size=encoder_hidden.size(1)
        #2.準備decoder第一個時間步的輸入, [batch_size,1] sos 作為輸入
        decoder_input= torch.LongTensor(torch.ones([batch_size,1],dtype=torch.int64)*config.num_sequence.SOS).to(config.device)
        #3.在第一個時間步進行計算, 得到第一個時間步的輸出,hidden_state
        #4.把前一個時間步的輸出進行計算 得到第一個最終的輸出結果
        #5.把前一次的hidden_state作為當前時間步的hidden_state輸入, 把前一次的輸出作為當前時間步的輸入
        #循環4.5

        #保存預測的結果
        decoder_outputs=torch.zeros([batch_size,config.max_len+2,len(config.num_sequence)]).to(config.device)

        for t in range(config.max_len+2):
            decoder_output_t,decoder_hidden=self.forward_step(decoder_input,decoder_hidden)
            #保存decoder_output_t到decoder_outputs中
            decoder_outputs[:,t,:]= decoder_output_t

            value,index= torch.topk(decoder_output_t,1)
            # print(decoder_output_t.size())
            decoder_input=index


        return decoder_outputs,decoder_hidden
    def forward_step(self,decoder_input,decoder_hidden):
        '''
        計算時間步上的結果
        :param decoder_input:[batch_size,1]
        :param decoder_hidden:[1,batch_size,hidden_size]
        :return:
        '''
        # print('decoder_in_size:',decoder_input.size())
        decoder_input_embedded=self.embedding(decoder_input)
        # print(decoder_input_embedded.size())
        #out:[batch_size,1,hidden_size] 1:seq_len
        #decoder_hidden:[1,batch_size,hidden_size]
        out,decoder_hidden=self.gru(decoder_input_embedded,decoder_hidden) #夢裡尋他千百度

        out= out.squeeze(1) #[batch_size,hidden_size]
        output=F.log_softmax(self.fc(out),dim=-1) #[batch_size,vocab_size]
        # print("output:",output.size())

        return output,decoder_hidden

    def evaluate(self,encoder_hidden):
        #評估
        decoder_hidden= encoder_hidden  #[1,batch_size,hidden_size]
        batch_size=encoder_hidden.size(1)
        decoder_input= torch.LongTensor(torch.ones([batch_size,1],dtype=torch.int64)*config.num_sequence.SOS).to(config.device)

        # print('decoder_input:',decoder_input.size())
        # print('decoder_input:',decoder_input)
        # print('decoder_hidden:',decoder_hidden)
        indices=[]
        # while True:
        for i in range(config.max_len+3):
            #decoder_output_t:[batch_size,vocab_size]
            decoder_output_t,decoder_hidden= self.forward_step(decoder_input,decoder_hidden)
            # print('decoder_outout_t:',decoder_output_t)
            # print('decoder_out_size:',decoder_output_t.size())
            value,index= torch.topk(decoder_output_t,1) #[batch_size,1]
            # print('index',index.size())
            decoder_input=index
            # if index.item() == config.num_sequence.EOS:
            #     break
            indices.append(index.squeeze(-1).cpu().detach().numpy())
            # print(indices.__len__())

        return indices