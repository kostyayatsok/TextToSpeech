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
        length_diff = batch["spectrogram"].size(-1) -\
                          batch["spectrogram_pred"].size(-1)
        batch["spectrogram_pred"] = F.pad(
            batch["spectrogram_pred"], (0, length_diff, 0, 0, 0, 0),
            "constant", self.mel_silence_value)
        mel_loss = self.criterion_mel(
            batch["spectrogram_pred"], batch["spectrogram"])
        
        batch["rel_durations_pred"] = batch["durations_pred"] /\
                                          batch["spectrogram"].size(-1)
        duration_loss = self.criterion_duration(
            batch["rel_durations_pred"].float(), batch["rel_durations"].float())

        print(
            f"Total loss: {(mel_loss + duration_loss).item():.4f}",
            f"Mel loss: {mel_loss.item():.4f}",
            f"Dur loss: {duration_loss.item():.8f}")
        return mel_loss + duration_loss * self.duration_weight
