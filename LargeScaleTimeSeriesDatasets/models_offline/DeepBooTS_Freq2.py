import torch
import torch.nn as nn
from layers.RevIN import RevIN
from math import sqrt
import torch.nn.functional as F


class Layer(nn.Module):

    def __init__(self, configs, hidden_size=None):
        super(Layer, self).__init__()
        self.scale = 0.02
        self.hidden_size = hidden_size if hidden_size else configs.d_model
        
        self.w = nn.Parameter(self.scale * torch.randn(1, self.hidden_size))

        # self.fc = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_size, self.hidden_size)
        # )
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(configs.dropout)
        )



    def fft(self, x):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(self.w.to(x.device), dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.hidden_size, dim=2, norm="ortho")
        return out

    def forward(self, x):
        x = self.fft(x)  # B, N, D
        return self.fc(x)

class Attention(nn.Module):

    def __init__(self, configs):
        super(Attention, self).__init__()
        
        self.layer = Layer(configs)
        self.H = configs.n_heads
        self.dropout = nn.Dropout(configs.dropout)
        
        
    def forward(self, x):
        B, L, E = x.shape
        queries = self.layer(x).view(B, L, self.H, -1)
        keys = self.layer(x).view(B, L, self.H, -1)
        values = self.layer(x).view(B, L, self.H, -1)
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values).contiguous()

        return V.view(B, L, -1), A


class ActLayer(nn.Module):
    def __init__(self, configs, in_channels=None):
        super(ActLayer, self).__init__()
        self.in_channels = in_channels if in_channels else configs.d_model
        d_ff = 4 * self.in_channels
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=configs.d_model, kernel_size=1)
        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.relu if configs.activation == "relu" else F.gelu
        
    def forward(self, x):
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        x = self.dropout(self.conv2(x).transpose(-1, 1))
        return x

class EncoderLayer(nn.Module):
    def __init__(self, configs):
        super(EncoderLayer, self).__init__()
        self.attention = Attention(configs)
        
        self.ln0 = ActLayer(configs)
        self.ln1 = ActLayer(configs)
        self.ln2 = ActLayer(configs)
        self.ln3 = ActLayer(configs, in_channels=configs.d_model*2)
        self.ln4 = ActLayer(configs, in_channels=configs.d_model*2)
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.relu if configs.activation == "relu" else F.gelu

    def forward(self, x):
        new_x, attn = self.attention(x)
        x = x + self.dropout(new_x)
        x_ln = x = self.norm1(x)
        x_ln = self.activation(self.ln0(x_ln))
        
        x = x - x_ln
        h = F.sigmoid(self.ln1(x)) * self.ln2(x)
        out =  torch.cat((new_x, x_ln),-1)
        out = F.sigmoid(self.ln3(out)) * self.ln4(out)
        
        return self.norm2(h), out

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.embed = nn.Linear(configs.seq_len, configs.d_model)
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.layers = nn.ModuleList([EncoderLayer(configs) for _ in range(configs.e_layers)])
        self.ln_out = nn.Linear(configs.d_model, configs.pred_len)

        print('deep pai ...')

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        z = x
        z = self.revin_layer(z, 'norm')
        x = z

        x = x.permute(0, 2, 1)
        
        x = self.embed(x)

        y_ = 0
        for layer in self.layers:
            x,out = layer(x)
            y_ = out - y_
        
        y_ = self.ln_out(y_)
        
        x = y_.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z

        return x