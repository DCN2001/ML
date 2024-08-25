import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.autograd import Variable

#Main model
class Trans_Attrc(nn.Module):
    def __init__(self, kernel_size=16, channels=256, chunk_size=250,n_heads=8, n_intra=4,n_inter=2,r=4):
        super().__init__()
        #Hyperparam 
        self.kernel_size = kernel_size
        self.channels = channels
        self.chunk_size = chunk_size
        self.hidden_units = channels  #LSTM unit
        #Layer
        self.encoder = Encoder(self.kernel_size, self.channels)
        #Layer Norm + Linear
        self.LayerNorm = nn.LayerNorm(self.channels)
        self.Linear1 = nn.Linear(in_features=self.channels, out_features=self.channels, bias=None)   #Without bias
        #DPTBlock
        self.DPTBlock = DPT_block(self.channels, n_heads,n_intra,n_inter)
        #LSTM EDA
        self.EDA = EDA(self.channels, self.hidden_units, r)

    def pad_segment(self, input, chunk_size):
        pad_device = input.device
        batch_size, dim, seq_len = input.shape
        overlap = chunk_size // 2      #overlap 50%
        pad_length = chunk_size - (overlap + seq_len % chunk_size) % chunk_size    #The length to be pad

        if pad_length > 0:
            pad = torch.zeros(batch_size, dim, pad_length, device=pad_device)
            input = torch.cat([input, pad], 2)

        pad_aux = torch.zeros(batch_size, dim, overlap, device=pad_device)
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, pad_length

    def chunking(self, input, chunk_size):
        input, pad_length = self.pad_segment(input, chunk_size)
        batch_size, dim, seq_len = input.shape
        overlap = chunk_size // 2
        segments1 = input[:, :, :-overlap].contiguous().view(batch_size, dim, -1, chunk_size)
        segments2 = input[:, :, overlap:].contiguous().view(batch_size, dim, -1, chunk_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, chunk_size).transpose(2, 3).contiguous()
        return segments, pad_length
    
    def criterion(self, output, num_instru):
        loss_sum = 0.0
        for i in range(output.shape[0]):
            #Create label that match the output shape but the number of 1 is equal to num_instru else 0
            device = output[i].device
            output_num = output[i].shape[1]
            label_num_attrc = torch.zeros((1,output_num), dtype=torch.float32, device=device)
            label_num_attrc[0, :num_instru[i]] = 1.0

            speaker_loss = F.binary_cross_entropy(output[i],label_num_attrc)
            loss = speaker_loss
            loss_sum += loss
        loss_avg = loss_sum/output.shape[0]
        return loss_avg

    def forward(self, x, num_instru):
        #Create zeros
        device = x.device
        max_num_instru = torch.max(num_instru)
        zeros = torch.zeros((x.shape[0],max_num_instru+1,self.hidden_units), device=device)
        #Encoder
        x = self.encoder(x)
        #Layer norm + Linear
        x = self.LayerNorm(x.permute(0, 2, 1).contiguous())
        x = self.Linear1(x).permute(0, 2, 1).contiguous()
        #Chunking
        x, pad_length = self.chunking(x, self.chunk_size) 
        #DPT block
        x = self.DPTBlock(x)
        #EDA
        x = self.EDA(x, zeros)
        #Estimating loss
        loss = self.criterion(x,num_instru)
        return loss

#Waveform encoder(Done)
class Encoder(nn.Module):
    def __init__(self, kernel_size, channels):
        super().__init__()
        #Hyper parmas
        self.kernel_size = kernel_size
        self.channels = channels
        #Layer
        self.Conv1d = nn.Conv1d(in_channels=2, out_channels=self.channels, kernel_size=self.kernel_size, stride=self.kernel_size//2, padding=0, bias=False)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.Relu(self.Conv1d(x))
        return x

#Dual path Transformer
class DPT_block(nn.Module):
    def __init__(self, channels, n_heads, n_intra, n_inter):
        super().__init__()
        #Hyper params
        self.n_intra = n_intra
        self.n_inter = n_inter 
        #Layer
        self.intra_PositionalEncoding = Pos_Encoding(d_model=channels, max_len=5000)
        self.intra_transformer = nn.ModuleList([])
        for i in range(self.n_intra):
            self.intra_transformer.append(TransformerEncoder(d_model=channels,nhead=n_heads,dropout=0.1))
        self.inter_PositionalEncoding = Pos_Encoding(d_model=channels, max_len=5000)
        self.inter_transformer = nn.ModuleList([])
        for i in range(self.n_intra):
            self.inter_transformer.append(TransformerEncoder(d_model=channels,nhead=n_heads,dropout=0.1))

    def forward(self, x):
        B, C, K, L = x.shape   #batch size, channls, chunk size, feature length
        #intra
        row_x = x.permute(0,3,2,1).contiguous().view(B*L,K,C)
        row_x1 = self.inter_PositionalEncoding(row_x)
        for i in range(self.n_intra):
            row_x1 = self.intra_transformer[i](row_x1.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        row_f = row_x1 + row_x
        row_out = row_f.view(B,L,K,C).permute(0,3,2,1).contiguous()
        #inter
        col_x = row_out.permute(0,2,3,1).contiguous().view(B*K,L,C)
        col_x1 = self.inter_PositionalEncoding(col_x)
        for i in range(self.n_inter):
            col_x1 = self.intra_transformer[i](col_x1.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        col_f = col_x1 + col_x
        col_out = col_f.view(B,L,K,C).permute(0,3,2,1).contiguous()
        return col_out

#Transformer
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dropout=0):
        super().__init__()
        self.LayerNorm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.Dropout1 = nn.Dropout(p=dropout)
        self.LayerNorm2 = nn.LayerNorm(normalized_shape=d_model)
        self.FeedForward = nn.Sequential(nn.Linear(d_model, d_model*2*2),
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout),
                                         nn.Linear(d_model*2*2, d_model))
        self.Dropout2 = nn.Dropout(p=dropout)
    def forward(self,x):
        x1 = self.LayerNorm1(x)
        x2 = self.self_attn(x1,x1,x1,attn_mask=None, key_padding_mask=None)[0]
        x3 = self.Dropout1(x2) + x    #Residual
        x4 = self.LayerNorm2(x3)
        x5 = self.FeedForward(x4)
        x6 = self.Dropout2(x5) + x3   #Residual 
        return x6 

class Pos_Encoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1323000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)  # seq_len, batch, channels
        pe = pe.transpose(0, 1).unsqueeze(0)  # batch, channels, seq_len
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        # x is seq_len, batch, channels
        # x = x + self.pe[:x.size(0), :]
        # x is batch, channels, seq_len
        x = x + self.pe[:, :, :x.size(2)]
        x = self.dropout(x)
        x = x.permute(0, 2, 1).contiguous()
        return x

#LSTM based encoder-decoder 
class EDA(nn.Module):
    def __init__(self,channels,hidden_units,r):
        super().__init__()
        #Hyper params
        self.n_units = hidden_units
        #Sequence aggregation
        self.LinearA = nn.Sequential(nn.Linear(channels,channels*2),
                                     nn.Tanh())
        self.LinearB = nn.Linear(channels,channels//r)
        self.LinearC = nn.Sequential(nn.Linear(channels*2,r),
                                     nn.Softmax(dim=-1))
        #EDA
        self.LSTM_encoder = nn.LSTM(input_size=self.n_units,hidden_size=self.n_units,num_layers=1,dropout=0.0,batch_first=True)
        self.LSTM_decoder = nn.LSTM(input_size=self.n_units,hidden_size=self.n_units,num_layers=1,dropout=0.0,batch_first=True)
        #Linear + sigmoid
        self.Linear_count = nn.Sequential(nn.Linear(self.n_units,1),
                                          nn.Sigmoid())
    def forward(self, x, zeros):
        #Sequence aggregation
        x = x.permute(0,3,2,1).contiguous()   #B,L,K,C
        x_a = self.LinearA(x)
        x_b = self.LinearB(x)
        x_c = self.LinearC(x_a)#.transpose(2,3)
        #print(x_b.shape,x_c.shape)   #torch.Size([1, 134, 250, 64]) torch.Size([1, 134, 250, 4])
        x_b_expanded = x_b.unsqueeze(3) 
        x_c_expanded = x_c.unsqueeze(4) 
        x_mult = x_b_expanded * x_c_expanded  
        x_agg = x_mult.mean(dim=2)  
        x_agg = x_agg.view(x_agg.size(0), x_agg.size(1), -1)  # reshape to num_chunks * channels
        #EDA
        self.LSTM_encoder.flatten_parameters()
        self.LSTM_decoder.flatten_parameters()
        x_agg = x_agg[:, torch.randperm(x_agg.size(1)), :]  #shuffle
        _, (hx,cx) = self.LSTM_encoder(x_agg)
        attractors, (_,_) = self.LSTM_decoder(zeros,(hx,cx))
        #Linear + Sigmoid
        prob = self.Linear_count(attractors).permute(0,2,1).contiguous()
        return prob