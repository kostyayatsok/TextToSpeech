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
from tqdm import tqdm

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
    featurizer = MelSpectrogram(config["MelSpectrogram"]).to(device)
    aligner = GraphemeAligner(config["MelSpectrogram"]).to(device)
    if config["wandb"]:
        wandb.init(project=config["wandb_name"])

    def process_batch(batch):
        batch["tokens"] = batch["tokens"].to(device)
        batch["waveform"] = batch["waveform"].to(device)
        batch["waveform_length"] = batch["waveform_length"].to(device)
        batch.update(featurizer(batch["waveform"], batch["waveform_length"]))
        batch['rel_durations'] = aligner(
            batch["waveform"], batch["waveform_length"], batch["transcript"]
        ).to(device)
        batch['durations'] = batch['rel_durations'] * batch['mel_length'][:, None]
        batch['durations'] = batch['durations'].ceil().long()
        batch['mask'] = torch.arange(batch['tokens'].size(-1))[None, :] < batch['tokens_length'][:, None]
        outputs = text2mel_model(**batch)
        batch.update(outputs)
        
        batch.update(criterion(batch))
        return batch
    step = 0
    for epoch in tqdm(range(config["n_epoch"])):
        for batch in train_loader:
            try:
                batch["epoch"] = epoch
                batch["step"] = step
                step += 1            
                optimizer.zero_grad()
                batch = process_batch(batch)
                batch["loss"].backward()
                optimizer.step()
                batch["loss"] = batch["loss"].item()
                batch["dur_loss"] = batch["dur_loss"].item()
                batch["mel_loss"] = batch["mel_loss"].item()
                break
            except Exception as inst:
                print(inst)
                pass
        if config["wandb"]:
            if config["log_audio"]:
                batch["wavform_pred"] = vocoder_model.inference(batch["mel_pred"]).cpu()
            else:
                batch["wavform_pred"] = batch["wavform"]
            log_batch(batch)

        # with torch.no_grad():
        #     val_loss = [[],[],[]]
        #     for batch in val_loader:
        #         try:
        #             batch = process_batch(batch)
        #             val_loss[0].append(batch["loss"])
        #             val_loss[1].append(batch["dur_loss"])
        #             val_loss[2].append(batch["mel_loss"])
        #         except Exception as inst:
        #             print(inst)
        #             pass
        #     batch["loss"] = torch.mean(val_loss[0])
        #     batch["dur_loss"] = torch.mean(val_loss[1])
        #     batch["mel_loss"] = torch.mean(val_loss[2])
        #     batch["wavform_pred"] = vocoder_model.inference(batch["mel_pred"]).cpu()
        #     batch["epoch"] = epoch
        #     batch["step"] = step
        #     step += 1           
        #     if config["wandb"]:
        #         log_batch(batch, mode="val")
        torch.save(text2mel_model.state_dict(), "text2mel_model.pt")

def log_batch(batch, mode="train"):
    idx = np.random.randint(batch["mel"].size(0))
    wandb.log({
        "step": batch["step"],
        "epoch": batch["epoch"],
        f"loss_{mode}": batch["loss"],
        f"dur_loss_{mode}": batch["dur_loss"],
        f"mel_loss_{mode}": batch["mel_loss"],
        f"original_mel_{mode}": wandb.Image(batch["mel"][idx].cpu().detach().numpy(), caption="original_mel"),
        f"pred_mel_{mode}": wandb.Image(batch["mel_pred"][idx].cpu().detach().numpy(), caption="mel_pred"),    
        f"original_audio_{mode}": wandb.Audio(batch["wavform"][idx].cpu().detach().numpy(), caption="original_audio", sample_rate=22050),
        f"pred_audio_{mode}": wandb.Audio(batch["wavform_pred"][idx].cpu().detach().numpy(), caption="pred_audio", sample_rate=22050),    
    })
if __name__ == "__main__":
    #copy-pasted from asr-homework-template
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