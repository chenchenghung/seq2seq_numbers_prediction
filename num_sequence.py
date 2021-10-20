

class Num_sequence:

    PAD_TAG = 'PAD'
    PAD=0
    UNK_TAG= 'UNK'
    UNK =1
    SOS_TAG= 'SOS'
    SOS=2
    EOS_TAG= 'EOS'
    EOS = 3
    def __init__(self):
        self.dict={self.PAD_TAG:self.PAD,
                   self.UNK_TAG:self.UNK,
                   self.SOS_TAG:self.SOS,
                   self.EOS_TAG:self.EOS
                   }
        for i in range(10):
            self.dict[str(i)]=len(self.dict)

        self.inverse_dict=dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentence,max_len,add_eos=False):
        #把sentence 轉化為數字序列
        #eg1:
        #sentence:11 , max_len:10
        #eg2:
        #sentence:8 , max_len:10
        #add_eos:true 輸出句子長度為max_len+1
        #add_eos:false 輸出的句子長度為max_len

        if len(sentence)> max_len: #句子的長度比max_len長
            sentence= sentence[:max_len]
        sentence_len=len(sentence)  #提前計算句子長度
        if add_eos:
            sentence= sentence+[self.EOS_TAG]

        if sentence_len< max_len:
            sentence = sentence + [self.PAD_TAG]*(max_len-sentence_len)
        result= [self.dict.get(i,self.UNK) for i in sentence]
        return result


    def inverse_transform(self,indices):
        #把序列轉化為原始的字符串數字
        return [self.inverse_dict.get(i,self.UNK_TAG) for i in indices]

    def __len__(self):
        return len(self.dict)
if __name__=='__main__':
    # num_sequence=Num_sequence()
    # print(len(num_sequence))
    # print(num_sequence.__len__())
    num_Sequence = Num_sequence()
    print(num_Sequence.dict)
    s = list("123123")
    ret = num_Sequence.transform(s,5)
    print(ret)
    ret = num_Sequence.inverse_transform(ret)
    print(ret)