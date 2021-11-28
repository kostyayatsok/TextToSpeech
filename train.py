import argparse
from tts.data_utils.Aligner import GraphemeAligner
from tts.utils import ConfigParser
from torch.utils.data import DataLoader, random_split
from tts.data_utils import LJSpeechCollator, LJSpeechDataset
from tts.models import FastSpeechModel, Vocoder
import torch
import numpy as np
import tts.loss
from tts.data_utils import MelSpectrogram
import wandb
from torchsummary import summary

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main(config):
    dataset = LJSpeechDataset(config["LJSpeechRoot"])
    if config["val_split"] > 0:
        val_len = int(len(dataset) * config["val_split"])
        train_len = len(dataset) - val_len
        print(f"Use {train_len} samples for training and {val_len} for validation.")
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size'],
            collate_fn=LJSpeechCollator()
        )
    else:
        train_dataset = dataset
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        collate_fn=LJSpeechCollator()
    )
    text2mel_model = FastSpeechModel(config["FastSpeech"]).to(device)
    # summary(text2mel_model, [(100,), (100,)], device=device)
    print(f"Total model parameters: \
        {sum(p.numel() for p in text2mel_model.parameters())}")
    vocoder_model = Vocoder().to(device).eval()
    optimizer = config.init_obj(
        config["optimizer"], torch.optim, text2mel_model.parameters()
    )
    criterion = config.init_obj(config["loss"], tts.loss)
    featurizer = MelSpectrogram(config["MelSpectrogram"])
    aligner = GraphemeAligner(config["MelSpectrogram"]).to(device)
    if config["wandb"]:
        wandb.init(config["wandb_name"])

    def process_batch(batch):
        batch["waveform"] = batch["waveform"].to(device)
        batch["waveform_length"] = batch["waveform_length"].to(device)
        batch["tokens"] = batch["tokens"].to(device)
        
        batch.update(featurizer(batch["waveform"], batch["waveform_length"]))
        batch['rel_durations'] = aligner(
            batch["waveform"], batch["waveform_length"], batch["transcript"]
        )
        batch['durations'] = (batch['rel_durations'] * batch['mel_length'])
        batch['durations'] = batch['durations'].ceil().long()
        
        outputs = text2mel_model(**batch)
        batch.update(outputs)
        
        batch["loss"] = criterion(batch)
        return batch
    step = 0
    for epoch in range(10000):
        for batch in train_loader:
            batch["epoch"] = epoch
            batch["step"] = step
            step += 1            
            optimizer.zero_grad()
            batch = process_batch(batch)
            batch["loss"].backward()
            optimizer.step()
            break     
        # batch["wavform_pred"] = vocoder_model.inference(batch["mel_pred"]).cpu()
        if config["wandb"]:
            log_batch(batch)    
        # for batch in val_loader:
        #     batch["epoch"] = epoch
        #     batch["step"] = step
        #     step += 1            
        #     batch = process_batch(batch)
        
def log_batch(batch):
    idx = np.random.randint(batch["mel"].size(0))
    wandb.log({
        "loss": batch["loss"],
        "step": batch["step"],
        "epoch": batch["epoch"],
        "original_mel": wandb.Image(batch["mel"][idx].detach().numpy(), caption="original_mel"),
        "pred_mel": wandb.Image(batch["mel_pred"][idx].detach().numpy(), caption="mel_pred"),    
        # "original_audio": wandb.Audio(batch["wavform"].detach().numpy(), caption="original_audio", sample_rate=22050),
        # "pred_audio": wandb.Audio(batch["wavform_pred"].detach().numpy(), caption="pred_audio", sample_rate=22050),    
    })
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args)
    
    main(config)