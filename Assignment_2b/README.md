# Assignment 2b. Multimodal emotion recognition on TESS

Three emotion classifiers for the [Toronto emotional speech set](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess),
one from speech only, one from text only and one that fuses both, each built from the functional blocks of the
assignment (preprocessing, feature extraction, temporal or contextual modelling, fusion, classifier).

## The data, and what it means for the text branch

TESS has 2,800 clips. Two actresses (OAF, aged 64, and YAF, aged 26) each say "Say the word ..." with 200 target
words, every word in all 7 emotions (angry, disgust, fear, happy, neutral, pleasant surprise, sad). The text of a
clip is therefore the same for all 7 emotions, and a text only model cannot do better than chance (14.3 %) on
transcripts. I keep that model as the assignment asks and use it as a control, and I add a second text source,
the Whisper transcript of each clip, to test whether text from speech recognition carries any emotional cue.

## Splits

| split | train | val | test | what it tests |
|---|---|---|---|---|
| `word` (main) | 1,960 clips, 140 words | 420 clips, 30 words | 420 clips, 30 words | new words, both speakers seen |
| `cross_speaker` | 1,190 OAF clips, 170 words | 210 OAF clips, 30 words | all 1,400 YAF clips | a new voice |

Each word sits in only one part of the word split, so no model can recognise a test clip by its word.
`data/splits.json` holds the exact lists (seed 1234).

## Architecture

| block | speech | text |
|---|---|---|
| preprocessing | resample to 16 kHz, trim silence 30 dB below peak (50 ms kept), cap at 4 s, no loudness normalisation | Unicode and whitespace cleanup, WordPiece tokens (bert-base-uncased) |
| feature extraction (frozen) | WavLM Base+ hidden states of all 13 layers, T x 768 at 50 frames/s | BERT base hidden states of all 13 layers, N x 768 |
| layer mixing (learned) | softmax weighted sum of the 13 layers | softmax weighted sum of the 13 layers |
| temporal / contextual modelling | projection to 256, 2 layer BiLSTM (2 x 128), attentive pooling | projection to 256, 2 layer Transformer encoder (4 heads), attentive pooling |
| fusion | gated fusion of the two 256 d vectors (concatenation as a baseline) | |
| classifier | LayerNorm, MLP 256 to 128 to 7 | |

A log mel baseline (80 bands, two 1-D convolutions before the same BiLSTM) shows what the pretrained speech
features add. Training uses AdamW (learning rate 1e-3, weight decay 0.01, one cycle schedule), label smoothing 0.05,
batch 32, up to 40 epochs with early stopping on validation macro F1 (patience 10), and three seeds per variant.

## Layout

```
models/
  speech_pipeline/train.py, test.py
  text_pipeline/train.py, test.py
  fusion_pipeline/train.py, test.py
  common/        data.py (dataset), features.py (preprocessing and feature extraction), nets.py (the blocks),
                 engine.py (training and testing loops)
scripts/
  prepare_tess.py         manifest and splits
  transcribe_whisper.py   Whisper transcripts (the second text source)
  run_experiments.py      every variant, split and seed
  analyse.py              tables, t-SNE, confusion matrices, failure cases
data/manifest.csv, splits.json, asr_whisper.json
Results/
  accuracy_tables.md / .csv    all model variants
  per_emotion.md, fusion_vs_speech.md, failure_cases.md, silhouette_scores.json
  plots/                       t-SNE of the temporal, contextual and fusion blocks, confusion matrices,
                               training curves, layer weights, failure case spectrograms
  runs/<run>/                  metrics.json, predictions.csv, train_log.csv, confusion_matrix.png, embeddings.npz
README.md
requirements.txt
```

## Running

```bash
pip install -r requirements.txt
# unzip the Kaggle download so the speaker and emotion folders sit in data/TESS/
python scripts/prepare_tess.py
python -m models.common.features --kind wav
python -m models.common.features --kind mel
python -m models.common.features --kind wavlm
python -m models.common.features --kind bert --text_source gt
python scripts/transcribe_whisper.py
python -m models.common.features --kind bert --text_source asr

python models/speech_pipeline/train.py --split word --seed 0       # one run
python models/speech_pipeline/test.py  --split word --seed 0
python scripts/run_experiments.py                                   # all 42 runs
python scripts/analyse.py
```

Every train and test script takes `--help`. The main options are `--split word|cross_speaker`, `--features
wavlm|mel` (speech and fusion), `--text_source gt|asr` (text and fusion) and `--fusion gated|concat`.
[colab/Assignment_2b_emotion.ipynb](../colab/Assignment_2b_emotion.ipynb) runs all of it on a Colab GPU.
