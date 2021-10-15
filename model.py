import torch
from torch import nn
#  https://github.com/sooftware/conformer/blob/main/conformer/encoder.py
from conformer.encoder import ConformerBlock
from conformer.modules import Linear
import torch.nn.functional as F
import os, sys


from torchinfo import summary

import math

class PositionalEncoding(nn.Module):

    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
        pe[:, 0::2] = torch.sin(position/denominator)
        pe[:, 1::2] = torch.cos(position/denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)


    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0],:,:]
        return outputBatch


# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)




class AVSRwithConf2(nn.Module): #with log_mel
    def __init__(self, numClasses):
        super(AVSRwithConf2, self).__init__()
        self.e = 17
        self.d_k = 256
        self.logmel_dim = 80
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_encode = PositionalEncoding(dModel=self.d_k, maxLen=1000)
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.d_k, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(self.d_k, self.d_k, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.input_projection = nn.Sequential(
            Linear(self.d_k * (((self.logmel_dim - 1) // 2 - 1) // 2), self.d_k),
            self.pos_encode,
            nn.Dropout(p=0.1),
        )
        self.linear1 = nn.Linear(512, self.d_k)
        self.layers_A = nn.ModuleList([ConformerBlock(encoder_dim=self.d_k).to(self.device) for _ in range(self.e)])
        self.linear3 = nn.Linear(256, 4 * self.d_k)
        self.linear4 = nn.Linear(1024, self.d_k)
        self.bn1 = nn.BatchNorm1d(num_features=4 * self.d_k)
        self.embeddings = nn.Embedding(numClasses, self.d_k)
        self.relu = nn.ReLU()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_k, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.linear5 = nn.Linear(256, numClasses-1)
        self.linear6 = nn.Linear(256, numClasses-1)
        

    def forward(self, x, tgt, tgt_mask, tgt_padding_mask):
        # x (Batch, T, 80)
        # tgt (Batch, S)
        # tgtmask (S, S)
        # tgt_paddingmask (Batch, S)
    
        # Conv2Dsubsampling  https://github.com/sooftware/conformer/blob/main/conformer/convolution.py
        x = self.sequential(x.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = x.size()
        x = x.permute(0, 2, 1, 3) 
        x = x.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim) 
        x = self.input_projection(x)


        #tgt embedding module
        tgt = self.embeddings(tgt) #tgt=(B, T), embedding=(B,T,D)
        tgt = tgt.transpose(0,1) #(T,B,D)
        tgt = self.pos_encode(tgt)

        #conformer block    
        for layer in self.layers_A:
            x = layer(x)

        # MLP
        x = self.linear3(x) # N,T,C
        x = x.transpose(1,2)
        x = self.bn1(x) # N,C,T
        x = x.transpose(1,2)
        x = self.relu(x)
        x = self.linear4(x)

        # to CE
        to_CE = x.transpose(0,1)
        to_CE = self.decoder(tgt=tgt, memory=to_CE, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        to_CE = self.linear5(to_CE)

        # to CTC
        x = self.linear6(x)
        x = F.log_softmax(x, dim=2) 

        to_CE = to_CE.transpose(0,1)
        return to_CE, x #(B,T=len(tgt), C),(B, T=len(frame), C)

    def encode(self, x):
    
        # logmel subsampling, linear, dropout
        x = self.sequential(x.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = x.size()
        x = x.permute(0, 2, 1, 3) #(N, T, encoderdim, 19) 19はdimの80が畳みこまれた結果。固定値
        x = x.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim) #(N, T/4, encoderdim x Dim) Dim=80
        x = self.input_projection(x) #(N, T/4, encoderdim)

        # backend embedding module
        x = self.linear1(x)
        x = self.pos_encode(x)


        #conformer block      
        for layer in self.layers_A:
            x = layer(x)

        # concat and MLP
        x = torch.cat([x, x], 2)
        x = self.linear3(x) # N,T,C
        x = x.transpose(1,2)
        x = self.bn1(x) # N,C,T
        x = x.transpose(1,2)
        x = self.relu(x)
        x = self.linear4(x)

        return x


class CustomLoss(nn.Module):
    def __init__(self, ramda, beta):
        super().__init__()
        self.ramda = ramda
        self.bata = beta
        self.loss_ctc = nn.CTCLoss(zero_infinity=True)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, CE_x, CTC_x, tgt, in_len, tgt_len):
        tgt = tgt.to(torch.long)
        CTC_x = CTC_x.transpose(0,1) #(T, B, C)
        loss1 = self.ramda*(self.loss_ctc(CTC_x, tgt, in_len, tgt_len))
        CE_x = CE_x.permute(0,2,1) #(B, C, T)
        loss2 = (1-self.ramda)*(self.loss_ce(CE_x, tgt))
        loss = loss1 + loss2

        return loss, loss1, loss2


if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    M2 = AVSRwithConf2(41)

    # avsrconf2 summary
    print(summary(M2, input_size=[(2, 160, 80), (2, 30), (30, 30), (2, 30)], dtypes=[torch.float, torch.long, torch.bool, torch.bool]))

