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
import torchaudio
class Trainer:
    def __init__(self, config):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(f"{config['save_dir']}/{self.run_id}/", exist_ok=True)
        write_json(config.config, f"{config['save_dir']}/{self.run_id}/config.json")
        
        self.tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self.train_loader, self.val_loader = build_dataloaders(config)
        print(f"Use {len(self.train_loader)} batches for training and",
                                        f"{0 if self.val_loader is None else len(self.val_loader)} for validation.")
            
        self.text2mel_model = FastSpeechModel(config["FastSpeech"]).to(self.device)
        print(f"Total model parameters: \
            {sum(p.numel() for p in self.text2mel_model.parameters())}")
        if config.resume is not None:
            print(f"Load text-to-mel model from checkpoint {config.resume}")
            self.text2mel_model.load_state_dict(torch.load(config.resume))

        self.vocoder = Vocoder().to(self.device).eval()

        self.optimizer = config.init_obj(
            config["optimizer"], torch.optim, self.text2mel_model.parameters()
        )
        self.scheduler = config.init_obj(
            config["scheduler"], torch.optim.lr_scheduler, self.optimizer,
            steps_per_epoch=len(self.train_loader), epochs=config["n_epoch"]
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
                    self.optimizer.zero_grad()
                    
                    batch = self.process_batch(batch)
                    self.metrics(batch)
                    batch["loss"].backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    batch["lr"] = self.scheduler.get_last_lr()[0]
                    
                    if self.config["wandb"] and\
                        self.step % self.config["wandb"] == 0:
                            self.log_batch(batch)
                    self.step += 1
                except Exception as inst:
                    print(inst)

            if self.config["wandb"]:
                self.log_batch(batch)
                    
            self.text2mel_model.eval()
            if self.val_loader is not None:
                with torch.no_grad():
                    for batch in self.val_loader:
                        try:
                            batch = self.process_batch(batch)
                            self.metrics(batch)
                        except Exception as inst:
                            print(inst)
                if self.config["wandb"]:
                    self.log_batch(batch, mode="val")
            if self.config["wandb"]:
                self.log_test()
            if self.config["save_period"] and\
                self.epoch % self.config["save_period"] == 0:
                    torch.save(
                        self.text2mel_model.state_dict(),
                        f"{self.config['save_dir']}/"+\
                        f"{self.run_id}/text2mel_model.pt"
                    )
        

    
    def process_batch(self, batch):
        #move tensors to cuda:
        for key in [
            "waveform", "waveform_length", "tokens", "token_lengths"]:
                batch[key] = batch[key].to(self.device)
            
        #creating mel spectrogram from ground truth wav
        batch.update(
            self.featurizer(batch["waveform"], batch["waveform_length"])
        )
        #get relative durations from aligner
        batch['rel_durations'] = self.aligner(
            batch["waveform"], batch["waveform_length"], batch["transcript"]
        ).to(self.device)
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
        mel = batch["mel"][idx].unsqueeze(0)
        mel_pred = batch["mel_pred"][idx].unsqueeze(0)
        wav = batch["waveform"][idx].cpu().detach().numpy()
        wav_rest = self.vocoder.inference(mel).cpu().detach().numpy()[0]
        wav_pred = self.vocoder.inference(mel_pred).cpu().detach().numpy()[0]
        mel = mel.squeeze(0).cpu().detach().numpy()
        mel_pred = mel_pred.squeeze(0).cpu().detach().numpy()
        
        dict2log = {
            "step": self.step,
            "epoch": self.epoch,
            f"loss_{mode}": self.metrics["loss"],
            f"dur_loss_{mode}": self.metrics["dur_loss"],
            f"mel_loss_{mode}": self.metrics["mel_loss"],
            f"orig_mel_{mode}": wandb.Image(mel, caption="Original mel"),
            f"pred_mel_{mode}": wandb.Image(mel_pred, caption="Predicted mel"),    
            f"orig_audio_{mode}": wandb.Audio(
                wav, caption="Original_audio", sample_rate=22050),
            f"rest_audio_{mode}": wandb.Audio(
                wav_rest,
                caption="Audio_from_original_mel", sample_rate=22050),
            f"pred_audio_{mode}": wandb.Audio(
                wav_pred,
                caption="Audio_from_predicted_mel", sample_rate=22050),
            f"text_{mode}": wandb.Html(batch["transcript"][idx]),
        }
        if "lr" in batch:
            dict2log["lr"] = batch["lr"]
        wandb.log(dict2log)

    @torch.no_grad()
    def log_test(self):
        self.text2mel_model.eval()
        for i, text in enumerate(self.config["TestData"]):
            tokens, length = self.tokenizer(text)
            tokens = tokens.to(self.device)
            mask = torch.ones(tokens.size(), dtype=torch.bool, device=self.device)
            mel = self.text2mel_model.inference(tokens, mask)
            wav = self.vocoder.inference(mel).cpu().detach().numpy()[0]
            mel = mel.cpu().detach().numpy()[0]
            wandb.log({
                "step": self.step,
                "epoch": self.epoch,
                f"Test/mel{i}": wandb.Image(mel, caption=text),
                f"Test/wav{i}": wandb.Audio(
                    wav, caption=text, sample_rate=22050),    
            })
            
    def lens2mask(self, max_len, lens):
        return torch.arange(max_len, device=lens.device)[None, :] <= lens[:, None]

