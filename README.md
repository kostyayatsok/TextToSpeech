# Text to speech
FastSpeech implementation for tts task. DLA CS HSE 2021 3rd homework by Kostya Elenik.

## Instalation guide
```console
git clone https://github.com/kostyayatsok/TextToSpeech.git
cd TextToSpeech
git clone https://github.com/NVIDIA/waveglow.git
pip install -qr requirements.txt
```
## Train
```console
python3 train.py -c configs/default_config.json
```

## Predict
```console
python3 test.py -o dir/to/save/output <input.txt
```

## Wandb
- report: https://wandb.ai/kostyayatsok/tts/reports/Text-to-speech-report--VmlldzoxMzAxMTI3
- all runs: https://wandb.ai/kostyayatsok/hifi?workspace=user-kostyayatsok
- best run: https://wandb.ai/kostyayatsok/hifi/runs/39lck09k
