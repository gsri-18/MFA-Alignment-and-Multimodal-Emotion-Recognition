# Assignment 2a. Forced alignment with the Montreal Forced Aligner

I align the six provided clips with [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io) 3
and compare two acoustic models, three pronunciation dictionaries and a speaker adapted model. One script runs
the whole pipeline, from the raw files to TextGrids, checks and figures.

| clips | source | speaker | length |
|---|---|---|---|
| F2BJRLP1, F2BJRLP2, F2BJRLP3 | BU Radio News Corpus (NIST SPHERE files) | f2b, a female news reader | 25 to 31 s each |
| ISLE_SESS0131_BLOCKD02_01 to 03 | ISLE corpus of non native English | one learner ("I said X not Y" minimal pairs) | about 4 s each |

## Setup

```bash
conda create -n mfa -c conda-forge python=3.11 montreal-forced-aligner=3
conda activate mfa
pip install -r requirements.txt
```

Put the provided `wav/` and `transcripts/` folders in `Assignment_2a/data/`. The audio is not in this repository
because both corpora are licensed.

## Running

```bash
cd Assignment_2a
python align.py                      # every stage, about 3 minutes on 4 CPU cores
python align.py --stage prepare      # or dictionaries | align | analyse | plot
```

The stages call MFA like this (paths shortened).

```bash
mfa g2p dictionaries/oov/words.txt english_us_arpa dictionaries/oov/g2p.dict            # missing words
mfa adapt corpus dictionaries/arpa.dict english_us_arpa outputs/models/english_us_arpa_adapted.zip
mfa align corpus dictionaries/arpa.dict english_us_arpa outputs/textgrids/arpa \
    --clean --beam 100 --retry_beam 400 --output_format long_textgrid
```

[colab/Assignment_2a_alignment.ipynb](../colab/Assignment_2a_alignment.ipynb) runs the same script on a Colab CPU
runtime, with MFA in a micromamba environment.

## What each stage does

**prepare** reads the SPHERE and WAV files with soundfile and writes 16 kHz 16 bit RIFF WAV, one folder per
speaker (`corpus/bu_f2b`, `corpus/isle_sess0131`) so MFA adapts to each speaker separately. The speaker of the BU
clips comes from the SPHERE header. Transcripts are normalised to the words that were spoken.

| in the transcript | aligned as | why |
|---|---|---|
| `S.J.C.'s` | `s j c's` | dotted acronyms are read letter by letter |
| `WBUR's` | `w b u r's` | an all capital word in mixed case text is a call sign, read letter by letter |
| `seventy- six`, `de-politicize` | `seventy six`, `de politicize` | hyphens split words |
| punctuation, case | removed, lower case | the dictionaries are lower case words |

**dictionaries** builds three lexicons for the 151 distinct words.

| name | phones | built from |
|---|---|---|
| `arpa` | ARPAbet | the `english_us_arpa` dictionary, and the `english_us_arpa` G2P model for the 4 missing words |
| `g2p_en` | ARPAbet | every word through g2p_en (all CMUdict variants, its neural G2P otherwise), as in my Assignment 1 |
| `mfa_ipa` | MFA IPA | the `english_mfa` dictionary, and the `english_us_mfa` G2P model for the 9 missing words |

**align** runs four configurations.

| run | acoustic model | dictionary |
|---|---|---|
| `arpa` (reference) | `english_us_arpa` 3.0.0 | `arpa` |
| `g2p_en` | `english_us_arpa` 3.0.0 | `g2p_en` |
| `mfa_ipa` | `english_mfa` 3.1.0 | `mfa_ipa` |
| `arpa_adapted` | `english_us_arpa` adapted to these clips with `mfa adapt` | `arpa` |

All model versions are pinned. The `english_mfa` dictionary stays at 3.0.0 because the 3.1.0 release file holds a
reduced lexicon with Indian English phones (`court` is `k ɒː ʈ`). Models are downloaded straight from the
[mfa-models releases](https://github.com/MontrealCorpusTools/mfa-models/releases), which avoids the GitHub API
limit of 60 calls per hour that `mfa model download` runs into on shared machines.

**analyse** writes the word and phone intervals of every run as CSV, checks every clip and compares each run with
the reference run, boundary by boundary.

* words that do not match the transcript, and `spn` phones (unknown words)
* phones at the 30 ms floor (three 10 ms frames, the shortest an MFA phone can be), which often marks a phone that
  was not pronounced
* speech phones longer than 250 ms, which usually absorb a pause or noise
* pauses inside the utterance, speech rate, and MFA's own log likelihood per clip

**plot** draws Praat style figures with Praat's own analysis through parselmouth (waveform, spectrogram, F0, word and
phone tiers), for every clip and for every run over the same window.

## Outputs

```
outputs/textgrids/<run>/<speaker>/<clip>.TextGrid    open in Praat together with the WAV in corpus/
outputs/analysis/summary.md                          the checks and the run comparison as tables
outputs/analysis/summary.json, <run>_words.csv, <run>_phones.csv, dictionaries.json, corpus.json
outputs/figures/                                     <clip>.png, <clip>_runs.png, and for the long clips
                                                     <clip>_overview.png and <clip>_zoom<k>.png
dictionaries/arpa.dict, g2p_en.dict, mfa_ipa.dict
```
