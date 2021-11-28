import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F

class FastSpeechLoss(nn.Module):
    def __init__(
        self, duration_weight, mel_silence_value,
        criterion_mel=MSELoss(), criterion_duration=MSELoss(),
        *args, **kwargs
    ):
        super().__init__()
        self.criterion_mel = criterion_mel
        self.criterion_duration = criterion_duration
        self.duration_weight = duration_weight
        self.mel_silence_value = mel_silence_value

    def forward(self, batch, *args, **kwargs):
        print(batch["spectrogram_pred"].shape, batch["spectrogram"].shape)
        length_diff = batch["spectrogram"].size(-1) -\
                          batch["spectrogram_pred"].size(-1)
        batch["spectrogram_pred"] = F.pad(
            batch["spectrogram_pred"], (0, length_diff, 0, 0, 0, 0),
            "constant", self.mel_silence_value)
        print(batch["spectrogram_pred"].shape, batch["spectrogram"].shape)
        
        mel_loss = self.criterion_mel(
            batch["spectrogram_pred"], batch["spectrogram"])
        duration_loss = self.criterion_duration(
            batch["durations_pred"].float(), batch["durations"].float())
        return mel_loss + duration_loss * self.duration_weight
