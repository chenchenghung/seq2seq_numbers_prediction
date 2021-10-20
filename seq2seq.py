'''
把encoder和decoder進行合併 得到seq2seq模型
'''
import torch.nn as nn
from encoder import Encoder
from decoder import  Decoder

class Seq2seq(nn.Module):
    def __init__(self):
        super(Seq2seq,self).__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()

    def forward(self,input,target,input_length,target_length):
        encoder_outputs,encoder_hidden= self.encoder(input,input_length)
        decoder_outputs,decoder_hidden= self.decoder(target,encoder_hidden)
        return decoder_outputs,decoder_hidden

    def evaluate(self,input,input_length):
        encoder_outputs,encoder_hidden= self.encoder(input,input_length)
        # print('encoder_h_size:',encoder_hidden.size())
        indices=self.decoder.evaluate(encoder_hidden)
        return indices