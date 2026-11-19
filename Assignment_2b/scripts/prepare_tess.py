""" Manifest and splits for the Toronto emotional speech set (TESS).

    python scripts/prepare_tess.py --tess_dir data/TESS

TESS has 2 speakers (OAF, aged 64, and YAF, aged 26) who each say the carrier phrase "Say the word ..."
with 200 target words in 7 emotions, 2,800 clips in all. The Kaggle copy holds every file twice (a
nested duplicate folder), so files are deduplicated by name. Speaker and emotion come from the folder
name, since one file is named OA_ instead of OAF_.

Writes data/manifest.csv (one row per clip) and data/splits.json with two splits
word            word disjoint. The 200 words are split 140 / 30 / 30 into train, val and test, and
                every clip of a word (both speakers, all emotions) goes to that word's part. Neither
                modality can rely on having seen the word.
cross_speaker   train and val on OAF (170 / 30 words), test on every YAF clip. An unseen voice.
"""

import argparse
import csv
import glob
import json
import os
import random

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
SEED = 1234


def emotion_of(folder):
    e = folder.split("_", 1)[1].lower()
    return "pleasant_surprise" if e.startswith("pleasant") else e


def scan(tess_dir):
    clips = {}
    for path in sorted(glob.glob(os.path.join(tess_dir, "**", "*.wav"), recursive=True)):
        name = os.path.basename(path)
        if name in clips:
            continue
        folder = os.path.basename(os.path.dirname(path))
        speaker, emotion = folder.split("_")[0].upper(), emotion_of(folder)
        parts = os.path.splitext(name)[0].split("_")
        word = "_".join(parts[1:-1]).lower()
        info = sf.info(path)
        clips[name] = {
            "utt_id": "{}_{}_{}".format(speaker, word, emotion),
            "path": os.path.relpath(path, ROOT), "speaker": speaker, "word": word, "emotion": emotion,
            "label": EMOTIONS.index(emotion), "text": "Say the word {}".format(word),
            "sample_rate": info.samplerate, "duration_s": round(info.duration, 3),
        }
    return sorted(clips.values(), key=lambda c: c["utt_id"])


def word_split(words, n_val, n_test, seed):
    words = sorted(words)
    random.Random(seed).shuffle(words)
    return set(words[n_val + n_test:]), set(words[:n_val]), set(words[n_val:n_val + n_test])


def make_splits(clips):
    words = {c["word"] for c in clips}
    tr, va, te = word_split(words, 30, 30, SEED)
    splits = {"word": {
        "train": [c["utt_id"] for c in clips if c["word"] in tr],
        "val": [c["utt_id"] for c in clips if c["word"] in va],
        "test": [c["utt_id"] for c in clips if c["word"] in te]}}
    tr, va, _ = word_split(words, 30, 0, SEED)
    splits["cross_speaker"] = {
        "train": [c["utt_id"] for c in clips if c["speaker"] == "OAF" and c["word"] in tr],
        "val": [c["utt_id"] for c in clips if c["speaker"] == "OAF" and c["word"] in va],
        "test": [c["utt_id"] for c in clips if c["speaker"] == "YAF"]}
    return splits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tess_dir", default=os.path.join(ROOT, "data", "TESS"))
    args = ap.parse_args()
    clips = scan(args.tess_dir)
    assert len(clips) == 2800, "expected 2800 TESS clips, found {}".format(len(clips))
    assert len({c["utt_id"] for c in clips}) == 2800
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(clips[0]))
        w.writeheader()
        w.writerows(clips)
    splits = make_splits(clips)
    with open(os.path.join(ROOT, "data", "splits.json"), "w") as f:
        json.dump(splits, f, indent=1)
    for name, s in splits.items():
        print(name, {k: len(v) for k, v in s.items()})
    rates = sorted({c["sample_rate"] for c in clips})
    print("sample rates", {r: sum(c["sample_rate"] == r for c in clips) for r in rates})
