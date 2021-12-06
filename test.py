import argparse
import sys
import os
import torch
import torchaudio 
from tts.data_utils import Vocoder
from tts.models import FastSpeechModel

fast_speech_config = {
    "n_phonemes": 42,
    "n_mels": 80,
    "n_encoder_blocks": 6,
    "n_decoder_blocks": 6,
    "hidden_size": 384,
    "Encoder":
    {
        "num_heads": 6,
        "kernel_size": 3,
        "filter_size": 1536,
        "dropout": 0.1
    },
    "Decoder":
    {
        "num_heads": 6,
        "kernel_size": 3,
        "filter_size": 1536,
        "dropout": 0.1
    },
    "LengthRegulator":
    {
        "kernel_size": 3,
        "dropout": 0.1
    }
}

weights_path = './FastSpeechWeights.pt'

def text2wav(text, path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    text2mel_model = FastSpeechModel(fast_speech_config).to(device)
    if not os.path.exists(weights_path):
        from google_drive_downloader import GoogleDriveDownloader as gdd
        gdd.download_file_from_google_drive(
            file_id='1Aeyq177z9hpEiK2FvWl-AK9nMQNwNV-Z',
            dest_path=weights_path
        )
    text2mel_model.load_state_dict(torch.load(weights_path))
    text2mel_model = text2mel_model.eval()
    vocoder = Vocoder().to(device).eval()
    tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
    
    tokens, length = tokenizer(text)
    tokens = tokens.to(device)
    mask = torch.ones(tokens.size(), dtype=torch.bool, device=device)
    mel = text2mel_model.inference(tokens, mask)
    wav = vocoder.inference(mel)[0].cpu().view(1, -1)
    
    torchaudio.save(path, wav, 22050)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args = args.parse_args()
    
    for text in sys.stdin:
        text2wav(text, f"{args.output}/{text}.wav")