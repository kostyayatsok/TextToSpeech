import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.utils.rnn import pad_sequence
import math

class ChannelFirstConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input):
        x = input
        x = x.transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2)
        return x
class FFTBlock(nn.Module):
    def __init__(self,
                 num_heads,
                 hidden_size,
                 kernel_size,
                 filter_size,
                 dropout):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_size)
        self.conv1d = nn.Sequential(
            ChannelFirstConv1d(
                hidden_size, filter_size, kernel_size=kernel_size, padding='same'), 
            nn.ReLU(),
            ChannelFirstConv1d(
                filter_size, hidden_size, kernel_size=kernel_size, padding='same'), 
        )
        self.norm_1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.norm_2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
    def forward(self, input):
        x, mask = input
        residual = x
        x = self.attention(x, mask)
        x = self.norm_1(x + residual)
        residual = x
        x = self.conv1d(x)
        x = self.norm_2(x + residual)
        return [x, mask]

class LengthRegulator(nn.Module):
    def __init__(self, hidden_size, kernel_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            ChannelFirstConv1d(
                hidden_size, hidden_size,
                    kernel_size=kernel_size, padding='same'),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            ChannelFirstConv1d(
                hidden_size, hidden_size, 
                    kernel_size=kernel_size, padding='same'),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.ReLU(), #length >= 1 ==> log(length) >= 0
        )
    def forward(self, input):
        durations = torch.exp(self.net(input))
        durations = durations.squeeze(-1)
        return durations

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        positions = torch.arange(max_len)
        #подсмотрел в каком-то туториале торча эту экспоненту
        frequancy = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        posisional_encondig = torch.zeros(max_len, d_model)
        posisional_encondig[:,0::2] =\
            torch.sin(positions[:,None]*frequancy[None,:])
        posisional_encondig[:,1::2] =\
            torch.cos(positions[:,None]*frequancy[None,:])
        self.register_buffer('posisional_encondig', posisional_encondig)
        
    def forward(self, x):
        return x + self.posisional_encondig[:x.size(1),:]

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d_model, d_model//n_heads) for _ in range(n_heads)])
        self.out = nn.Linear(d_model//n_heads*n_heads, d_model)
    def forward(self, inputs, mask):
        x = []
        for head in self.heads:
            x.append(head(inputs, mask))
        x = torch.cat(x, dim=-1)
        res = self.out(x)
        return res
        
class AttentionHead(nn.Module):
    def __init__(self, d_model, d_hid) -> None:
        super().__init__()
        self.key_fn = nn.Linear(d_model, d_hid)
        self.value_fn = nn.Linear(d_model, d_hid)
        self.query_fn = nn.Linear(d_model, d_hid)
        self.d_model = d_model
    def forward(self, inputs, mask):
        K = self.key_fn(inputs)
        Q = self.query_fn(inputs)
        V = self.value_fn(inputs)
        QK = torch.matmul(Q, K.transpose(-1, -2))/math.sqrt(self.d_model)
        if mask is not None:
            QK[mask] = -1e9
        QK = torch.softmax(QK, dim=-1)
        res = torch.matmul(QK, V)
        return res

class FastSpeechModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = nn.Embedding(
            config["n_phonemes"], config["hidden_size"])
        self.positional_encoding_1 = PositionalEncoding(config["hidden_size"])
        self.encoder = nn.Sequential(
            *[FFTBlock(hidden_size=config["hidden_size"], **config["Encoder"])\
                for _ in range(config["n_encoder_blocks"])]
        )
        self.length_regulator = LengthRegulator(
            hidden_size=config["hidden_size"], **config["LengthRegulator"])
        self.positional_encoding_2 = PositionalEncoding(config["hidden_size"])
        self.decoder = nn.Sequential(
            *[FFTBlock(hidden_size=config["hidden_size"], **config["Encoder"])\
                for _ in range(config["n_decoder_blocks"])]
        )
        self.linear_layer = nn.Linear(config["hidden_size"], config["n_mels"])
    
    def forward(self, tokens, mask, durations=None, mel_mask=None, *args, **kwargs):
        x = tokens
        x = self.embeddings(x)
        x = self.positional_encoding_1(x)
        x = self.encoder([x, mask])[0]
        
        durations_pred = self.length_regulator(x)
        if durations is None:
            durations = durations_pred.long()
        aligned, aligned_mask = [], []
        for one_input, one_mask, one_dur in zip(x, mask, durations):
            aligned.append(torch.repeat_interleave(one_input, one_dur.view(-1), dim=-2))
            aligned_mask.append(torch.repeat_interleave(one_mask, one_dur.view(-1), dim=-1))        
        aligned = pad_sequence(aligned, batch_first=True)
        if mel_mask is None:
            mel_mask = pad_sequence(aligned_mask, batch_first=True)
        else:
            mel_mask = F.pad(mel_mask, (0, aligned.size(-2)-mel_mask.size(-1), 0, 0))
        x = aligned
        x = self.positional_encoding_2(x)
        x = self.decoder([x, mel_mask])[0]
        mel = self.linear_layer(x)
        mel = mel.transpose(1, 2)
        return {
          "mel_pred": mel,
          "durations_pred": durations_pred
        }
    @torch.no_grad()
    def inference(self, tokens, mask):
        return self.forward(tokens, mask)["mel_pred"]