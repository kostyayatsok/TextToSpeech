import argparse
from tts.utils import ConfigParser
from torch.utils.data import DataLoader
from tts.data_utils import LJSpeechCollator, LJSpeechDataset
from tts.models import FastSpeechModel, Vocoder
import torch
import numpy as np
import tts.loss
from tts.data_utils import MelSpectrogram

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main(config):
    train_loader = DataLoader(
        LJSpeechDataset('.'), batch_size=config['batch_size'],
        collate_fn=LJSpeechCollator()
    )
    text2mel_model = FastSpeechModel(config["FastSpeech"])
    # vocoder_model = Vocoder()
    optimizer = config.init_obj(
        config["optimizer"], torch.optim, text2mel_model.parameters()
    )
    criterion = config.init_obj(config["loss"], tts.loss)
    featurizer = MelSpectrogram(config["MelSpectrogram"])

    for epoch in range(100):
        for batch in train_loader:
            batch['spectrogram'] = featurizer(batch["waveform"]) 
            batch['durations'] = featurizer(batch["waveform"]) 
            # batch.to(device)
            outputs = text2mel_model(**batch)
            batch.update(outputs)
            loss = criterion(batch)
            print(loss)
            
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