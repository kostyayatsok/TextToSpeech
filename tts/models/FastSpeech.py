import torch
from torch import nn

class FFTBlock(nn.Model):
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
            nn.Conv1d(hidden_size, filter_size, kernel_size=kernel_size), 
            nn.ReLU(),
            nn.Conv1d(filter_size, hidden_size, kernel_size=kernel_size), 
        )
        self.norm_1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(),
        )
        self.norm_2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(),
        )
    def forward(self, input):
        x, _ = self.attention(input, input, input, mask=None)
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
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size),
            nn.LayerNorm(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size),
            nn.LayerNorm(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.ReLU(), #length > 0
        )
    def forward(self, input, target_length):
        durations = torch.exp(self.net(input)).long()
        aligned = torch.repeat_interleave(input, durations, dim=-1)
        return durations, aligned

class FastSpeechModel(nn.Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = nn.Embedding(
            config["n_phonemes"], config["hidden_size"])
        self.positional_encoding_1 = lambda x : x #TODO: implement positional encoding
        self.encoder = nn.Sequential(
            *[FFTBlock(hidden_size=config["hidden_size"], **config["Encoder"])\
                for _ in range(config["n_encoder_blocks"])]
        )
        self.length_regulator = LengthRegulator(
            hidden_size=config["hidden_size"], **config["LengthRegulator"])
        self.positional_encoding_2 = lambda x : x #TODO: implement positional encoding
        self.decoder = nn.Sequential(
            *[FFTBlock(hidden_size=config["hidden_size"], **config["Encoder"])\
                for _ in range(config["n_decoder_blocks"])]
        )
        self.linear_layer = nn.Linear(config["hidden_size"], config["n_mels"]) #TODO: set correct in/out channels
    def forward(self, tokens, *args, **kwargs):
        x = self.embeddings(tokens)
        x = self.positional_encoding_1(x)
        x = self.encoder(x)
        durations, x = self.length_regulator(x)
        x = self.positional_encoding_2(x)
        x = self.decoder(x)
        spectrogram = self.linear_layer(x)
        return {"spectrogram_pred": spectrogram, "durations_pred": durations}