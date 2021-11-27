import torch.nn as nn
from torch import MSELoss

class FastSpeechLoss(nn.Module):
    def __init__(
        self, criterion_mel=MSELoss(), criterion_duration=MSELoss(),
            *args, **kwargs):
        super().__init__()
        self.criterion_mel = criterion_mel
        self.criterion_duration = criterion_duration
        self.duration_weight = 1.
    def forward(self, batch, *args, **kwargs):
        mel_loss = self.criterion_mel(
            batch["spectrogram_pred"], batch["spectrogram"])
        duration_loss = self.criterion_mel(
            batch["durations_pred"], batch["durations"])
        return mel_loss + duration_loss * self.duration_weight
