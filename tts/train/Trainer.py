from datetime import datetime
import numpy as np
import os
import torch
import tts.loss
import wandb
from tts.data_utils import MelSpectrogram, Vocoder
from tts.data_utils.Aligner import GraphemeAligner
from tts.data_utils.build_dataloaders import build_dataloaders
from tts.models import FastSpeechModel
from tts.utils import MetricsTracker
from tts.utils.util import write_json
from tqdm import tqdm

class Trainer:
    def __init__(self, config):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(f"{config['save_dir']}/{self.run_id}/", exist_ok=True)
        write_json(config.config, f"{config['save_dir']}/{self.run_id}/config.json")
        
        self.train_loader, self.val_loader = build_dataloaders(config)
        print(f"Use {len(self.train_loader)} samples for training and",
                                        f"{len(self.val_loader)} for validation.")
            
        self.text2mel_model = FastSpeechModel(config["FastSpeech"]).to(self.device)
        print(f"Total model parameters: \
            {sum(p.numel() for p in self.text2mel_model.parameters())}")
        self.vocoder = Vocoder().to(self.device).eval()

        self.optimizer = config.init_obj(
            config["optimizer"], torch.optim, self.text2mel_model.parameters()
        )
        
        self.criterion = config.init_obj(config["loss"], tts.loss)
        
        self.featurizer = MelSpectrogram(config["MelSpectrogram"]).to(self.device)
        
        self.aligner = GraphemeAligner(config["MelSpectrogram"]).to(self.device)
        
        if config["wandb"]:
            wandb.init(project=config["wandb_name"])
        self.metrics = MetricsTracker(["loss", "dur_loss", "mel_loss"])
        self.step = 0
        
        self.config = config
    
    def train(self):
        for self.epoch in tqdm(range(1, self.config["n_epoch"]+1)):
            self.text2mel_model.train()
            for batch in self.train_loader:
                try:
                    self.step += 1
                    self.optimizer.zero_grad()
                    
                    batch = self.process_batch(batch)
                    self.metrics(batch)
                    batch["loss"].backward()
                    self.optimizer.step()
                    if self.config["wandb"] and\
                        self.step % self.config["wandb"] == 0:
                            self.log_batch(batch)
                except Exception as inst:
                    print(inst)

            self.text2mel_model.eval()
            with torch.no_grad():
                for batch in self.val_loader:
                    try:
                        batch = self.process_batch(batch)
                        self.metrics(batch)
                    except Exception as inst:
                        print(inst)
                if self.config["wandb"]:
                    self.log_batch(batch, mode="val")
            self.log_test()
            if self.config["save_period"] and\
                self.epoch % self.config["save_period"] == 0:
                    torch.save(
                        self.text2mel_model.state_dict(),
                        f"{self.config['save_dir']}/\
                            {self.run_id}/weights/text2mel_model.pt"
                    )
        

    
    def process_batch(self, batch):
        #creating mel spectrogram from ground truth wav
        batch.update(
            self.featurizer(batch["waveform"], batch["waveform_length"])
        )
        #get relative durations from aligner
        batch['rel_durations'] = self.aligner(
            batch["waveform"], batch["waveform_length"], batch["transcript"]
        )
        #convert relative durations to absolute
        batch['durations'] = batch['rel_durations']*batch['mel_length'][:, None]
        batch['durations'] = batch['durations'].round().long()
        
        #create masks of padding
        batch['mask'] = self.lens2mask(
            batch['tokens'].size(-1), batch['token_lengths'])
        batch['mel_mask'] = self.lens2mask(
            batch['mel'].size(-1), batch['mel_length'])
        
        #run model
        outputs = self.text2mel_model(**batch)
        batch.update(outputs)

        #calculate loss
        batch.update(self.criterion(batch))
        return batch
    
    def log_batch(self, batch, mode="train"):
        idx = np.random.randint(batch["mel"].size(0))
        mel = batch["mel"][idx]
        mel_pred = batch["mel_pred"][idx]
        wav = batch["waveform"][idx].cpu().detach().numpy()
        wav_rest = self.vocoder.inference(mel).cpu()
        wav_pred = self.vocoder.inference(mel_pred).cpu()
        mel = mel.cpu().detach().numpy()
        mel_pred = mel_pred.cpu().detach().numpy()
        
        wandb.log({
            "step": self.step,
            "epoch": self.epoch,
            f"loss_{mode}": self.metrics["loss"],
            f"dur_loss_{mode}": self.metrics["dur_loss"],
            f"mel_loss_{mode}": self.metrics["mel_loss"],
            f"orig_mel_{mode}": wandb.Image(mel, caption="Original mel"),
            f"pred_mel_{mode}": wandb.Image(mel_pred, caption="Predicted mel"),    
            f"orig_audio_{mode}": wandb.Audio(
                wav, caption="Original audio", sample_rate=22050),
            f"rest_audio_{mode}": wandb.Audio(
                wav_rest,
                caption="Audio from original mel", sample_rate=22050),
            f"pred_audio_{mode}": wandb.Audio(
                wav_pred,
                caption="Audio from predicted mel", sample_rate=22050),    
        })

    @torch.no_grad()
    def log_test(self):
        self.text2mel_model.eval()
        for i, text in enumerate(self.config["TestData"]):
            tokens, length = self.tokenizer(text)
            mask = self.lens2mask(length)
            mel = self.text2mel_model.inference(tokens, mask)
            wav = self.vocoder.inference(mel).cpu()
            wandb.log({
                "step": self.step,
                "epoch": self.epoch,
                f"Test/mel{i}": wandb.Image(mel, caption=text),
                f"Test/wav{i}": wandb.Audio(
                    wav, caption=text, sample_rate=22050),    
            })
            
    def lens2mask(self, max_len, lens):
        return torch.arange(max_len)[None, :] < lens[:, None]

