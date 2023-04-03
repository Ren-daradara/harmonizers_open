import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        
        self.rnn = nn.LSTM(input_size=13, 
                    hidden_size = 73 , 
                    num_layers=2,
                    batch_first=True, 
                    dropout=0.2,
                    bidirectional=True)
        self.softmax=nn.Softmax(dim=-1)

            
    def forward(self, melody):
        out, hc=self.rnn(melody)
        out_front=out[:,:,0:73]
        out_behind=out[:,:,73:]
        out_sum=out_front+out_behind
        out_softmax=self.softmax(out_sum)
        #out_argmax=torch.argmax(out_softmax,dim=-1)
        return out_softmax