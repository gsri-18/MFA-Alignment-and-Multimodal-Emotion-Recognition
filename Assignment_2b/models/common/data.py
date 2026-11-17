""" Manifest, splits and the dataset that serves cached features to the three pipelines. """

import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]


def load_manifest():
    with open(os.path.join(ROOT, "data", "manifest.csv")) as f:
        return list(csv.DictReader(f))


def load_split(name):
    with open(os.path.join(ROOT, "data", "splits.json")) as f:
        return json.load(f)[name]


class TessFeatures(Dataset):
    """Cached [layers, time, dim] arrays for the speech features and/or the text features."""

    def __init__(self, utt_ids, speech=None, text_source=None, limit=0):
        from models.common.features import cache_dir
        clips = {c["utt_id"]: c for c in load_manifest()}
        self.items = [clips[u] for u in utt_ids][:limit or None]
        self.speech_dir = cache_dir(speech) if speech else None
        self.text_dir = cache_dir("bert", text_source) if text_source else None
        self.memory = {}

    def __len__(self):
        return len(self.items)

    def load(self, folder, utt_id):
        """Arrays stay in memory after the first epoch (about 4 GB for all WavLM features in float16).
        With DataLoader workers every worker keeps its own copy, so the default is no workers."""
        key = (folder, utt_id)
        if key not in self.memory:
            self.memory[key] = torch.from_numpy(np.load(os.path.join(folder, utt_id + ".npy")))
        return self.memory[key]

    def __getitem__(self, i):
        c = self.items[i]
        out = {"utt_id": c["utt_id"], "label": int(c["label"])}
        if self.speech_dir:
            out["speech"] = self.load(self.speech_dir, c["utt_id"])
        if self.text_dir:
            out["text"] = self.load(self.text_dir, c["utt_id"])
        return out


def pad(seqs):
    """[layers, T_i, D] tensors -> [B, layers, T_max, D] float32 and a [B, T_max] mask (True = real frame)."""
    n = max(s.shape[1] for s in seqs)
    x = torch.zeros(len(seqs), seqs[0].shape[0], n, seqs[0].shape[2])
    mask = torch.zeros(len(seqs), n, dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, :, :s.shape[1]] = s.float()
        mask[i, :s.shape[1]] = True
    return x, mask


def collate(batch):
    out = {"utt_id": [b["utt_id"] for b in batch], "label": torch.tensor([b["label"] for b in batch])}
    for key in ("speech", "text"):
        if key in batch[0]:
            out[key], out[key + "_mask"] = pad([b[key] for b in batch])
    return out
