
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, random_split
from tts.data_utils import LJSpeechCollator, LJSpeechDataset

def build_dataloaders(config):
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
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        collate_fn=LJSpeechCollator()
    )
    return train_loader, val_loader
