# nasa-space-sounds (PyTorch)

Pipeline:
1) Fetch metadata -> data/raw/metadata/sounds.json + manifest JSONL
2) Download audio -> data/raw/audio/
3) Normalize audio -> 16kHz mono WAV in data/interim/audio_16k_mono/
4) Deterministic splits -> data/splits/{train,val,test}.txt
5) Log-mel features -> data/processed/melspec_16k/{id}.pt

## Setup (venv)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run the dataset build
```bash
python scripts/00_fetch_metadata.py --limit 200
python scripts/01_download_audio.py
python scripts/02_normalize_audio.py
python scripts/03_make_splits.py
python scripts/04_build_features.py
```

## Sanity-check training loop
```bash
python -m nasa_sounds.train --split train --batch-size 16
```