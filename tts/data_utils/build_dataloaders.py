
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, random_split
from tts.data_utils import LJSpeechCollator, LJSpeechDataset
import sys
import os

def build_dataloaders(config):
    if not os.path.exists('LJSpeech-1.1') and not os.path.exists(f"{config['LJSpeechRoot']}/LJSpeech-1.1"):
        os.system('wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2')
        os.system('tar -xjf LJSpeech-1.1.tar.bz2')
    dataset = LJSpeechDataset(config["LJSpeechRoot"])
    if config["val_split"] > 0:
        val_len = int(len(dataset) * config["val_split"])
        train_len = len(dataset) - val_len
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size'],
            collate_fn=LJSpeechCollator()
        )
    else:
        train_dataset = dataset
        val_loader=None
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        collate_fn=LJSpeechCollator(), shuffle=True, pin_memory=True,
        num_workers=3,
    )
    return train_loader, val_loader
