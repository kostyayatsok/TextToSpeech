# Text to speech
FastSpeech implementation for tts task. DLA CS HSE 2021 3rd homework by Kostya Elenik.

## Instalation guide
```console
git clone https://github.com/kostyayatsok/TextToSpeech.git
git clone https://github.com/NVIDIA/waveglow.git
pip install -qr requirements.txt
```
## Train
```console
python3 train.py -c configs/default_config.json
```

## Predict
```console
python3 test.py -o path/to/output <"sentence for synthesis"
python3 test.py -o path/to/output <input.txt
```