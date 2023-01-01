# RADTTS + iSTFTNet vocoder

🇺🇦 Join Ukrainian Text-to-Speech community: https://t.me/speech_synthesis_uk

Clone the code:

```bash
git clone https://github.com/egorsmkv/radtts-istftnet
cd radtts-istftnet
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download Ukrainian RADTTS and iSTFTNet models:

```bash
mkdir models
cd models

wget https://github.com/egorsmkv/radtts-istftnet/releases/download/v1.0/iSTFTNet-Vocoder-Lada.pt
wget https://github.com/egorsmkv/radtts-istftnet/releases/download/v1.0/RADTTS-Lada.pt
```

Then you can inference own texts by the following command:

```bash
python3 inference.py -c config_ljs_dap.json -r models/RADTTS-Lada.pt -t test_sentences.txt --vocoder_checkpoint_file models/iSTFTNet-Vocoder-Lada.pt -o results/
```
