import torch
from torch import nn
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
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True) #TODO: Custom MultiheadAttention implementation
        self.conv1d = nn.Sequential(
            ChannelFirstConv1d(hidden_size, filter_size, kernel_size=kernel_size, padding='same'), 
            nn.ReLU(),
            ChannelFirstConv1d(filter_size, hidden_size, kernel_size=kernel_size, padding='same'), 
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
        x, _ = self.attention(input, input, input)
        residual = input
        x = self.norm_1(x + residual)
        residual = x
        x = self.conv1d(x)
        x = self.norm_2(x + residual)
        return x

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
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1),:]
        return self.dropout(x)
    
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
    def forward(self, tokens, durations, *args, **kwargs):
        x = tokens
        x = self.embeddings(x)
        x = self.positional_encoding_1(x)
        x = self.encoder(x)
        
        durations_pred = self.length_regulator(x)
        
        aligned = []
        for one_input, one_dur in zip(x, durations):
            aligned.append(torch.repeat_interleave(one_input, one_dur.view(-1), dim=-2))        
        aligned = pad_sequence(aligned, batch_first=True)
        
        x = aligned
        x = self.positional_encoding_2(x)
        x = self.decoder(x)
        mel = self.linear_layer(x)
        mel = mel.transpose(1, 2)
        return {
          "mel_pred": mel,
          "durations_pred": durations_pred
        }