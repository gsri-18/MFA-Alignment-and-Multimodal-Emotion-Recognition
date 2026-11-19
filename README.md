# Forced alignment and multimodal emotion recognition

Assignment 2 for the Speech Processing Lab, LTRC, IIIT Hyderabad, in two parts.

| | 2a. Forced alignment | 2b. Multimodal emotion recognition |
|---|---|---|
| Task | align 6 clips at word and phone level with the Montreal Forced Aligner | classify 7 emotions from speech, from text and from both |
| Data | 3 BU Radio News paragraphs (one news reader) and 3 ISLE learner sentences | Toronto emotional speech set (TESS), 2,800 clips, 2 speakers |
| Method | MFA 3.4.4, `english_us_arpa` and `english_mfa` models, three lexicons built for the corpus words with G2P for gaps, a speaker-adapted model | frozen WavLM Base+ and BERT features, BiLSTM and Transformer heads, gated fusion |
| Main result | every word aligned with `english_us_arpa`, dictionary and adaptation shift word boundaries by under one frame on average, `english_mfa` fails on two learner clips | 99.8% (speech) and 99.9% (fusion) on unseen words, 65.6% (speech) on an unseen speaker, text at chance |
| Ran on | Colab CPU, 7.4 min including the MFA install | Colab A100, 42 runs |
| Report | [Report/Assignment_2a_Report.pdf](Report/Assignment_2a_Report.pdf) | [Report/Assignment_2b_Report.pdf](Report/Assignment_2b_Report.pdf) |
| Details | [Assignment_2a/README.md](Assignment_2a/README.md) | [Assignment_2b/README.md](Assignment_2b/README.md) |

## Repository layout

```
Assignment_2a/               forced alignment
  align.py                   the whole pipeline (prepare, dictionaries, align, analyse, plot)
  plots.py                   Praat-style figures
  dictionaries/              the three pronunciation dictionaries
  outputs/textgrids/<run>/   TextGrids of the four alignments
  outputs/analysis/          checks, run comparison, word and phone tables
  outputs/figures/           figures of every clip and every run
Assignment_2b/               emotion recognition, in the layout the assignment asks for
  models/{speech,text,fusion}_pipeline/train.py, test.py
  models/common/             dataset, feature extraction, network blocks, training loop
  scripts/                   data preparation, Whisper transcription, all experiments, analysis
  data/                      manifest, splits and Whisper transcripts (not the audio)
  Results/                   accuracy tables, per-emotion recall, failure cases, plots, every run
colab/                       the two notebooks as Colab ran them, with every output and cell time
colab/logs/                  the session logs and cell timings the notebooks saved
Report/                      both reports (PDF and LaTeX), their figures and the script that draws them
scripts/                     Colab data bundling and the notebook generator
```

## Running

Each part has its own `requirements.txt` and README with the full commands.

```bash
# 2a (needs the provided clips in Assignment_2a/data/)
conda create -n mfa -c conda-forge python=3.11 montreal-forced-aligner=3.4.4
conda activate mfa && pip install -r Assignment_2a/requirements.txt
(cd Assignment_2a && python align.py)

# 2b (needs the TESS clips in Assignment_2b/data/TESS/)
pip install -r Assignment_2b/requirements.txt
(cd Assignment_2b && python scripts/prepare_tess.py)    # then features, experiments and analysis, see its README
```

Both notebooks in `colab/` ran unattended. They fetched the code (commit `b8e0c73`) and the data from a private
Hugging Face dataset repo, saved the results, the executed notebook and the logs to Hugging Face and Google Drive, and
then deleted their runtime. The 2b results came from two sessions. The first finished the word split and ended, and
the second restored those runs and did the rest. To rerun the work, use the commands in each part's README.

The audio is not in this repository. The BU Radio News and ISLE clips are licensed corpora, and TESS is on
[Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) under CC BY-NC 4.0.

## Acknowledgements

- Montreal Forced Aligner and its pretrained models, McAuliffe et al., Interspeech 2017.
- TESS, Pichora-Fuller and Dupuis, University of Toronto, 2020.
- WavLM (Chen et al., 2022), BERT (Devlin et al., 2019) and Whisper (Radford et al., 2023) through Hugging Face transformers.
- g2p_en, praat-parselmouth and tgt.
