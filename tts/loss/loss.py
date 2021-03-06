import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
import torch

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
        length_diff = batch["mel"].size(-1) - batch["mel_pred"].size(-1)
        batch["mel_pred"] = F.pad(
            batch["mel_pred"], (0, length_diff, 0, 0, 0, 0),
            "constant", self.mel_silence_value)
        mel_mask = torch.repeat_interleave(batch['mel_mask'], repeats=batch["mel"].size(1), dim=0).view(batch["mel"].size())
        mel_loss = self.criterion_mel(
            batch["mel_pred"][mel_mask], batch["mel"][mel_mask])
        
#         batch["rel_durations_pred"] =\
#             batch["durations_pred"] / batch["mel_length"]
        duration_loss = self.criterion_duration(
            batch["durations_pred"][batch["mask"]].float(), batch["durations"][batch["mask"]].float())
        return {
            "loss": mel_loss + duration_loss * self.duration_weight,
            "mel_loss": mel_loss,
            "dur_loss": duration_loss
        }
