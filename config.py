from num_sequence import Num_sequence
import torch
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_sequence= Num_sequence()

train_batch_size=256

max_len=9

embedding_dim=100

num_layer=1
hidden_size=64

model_save_path='model/seq2seq.model'
optimizer_save_path='model/optimizer.model'



