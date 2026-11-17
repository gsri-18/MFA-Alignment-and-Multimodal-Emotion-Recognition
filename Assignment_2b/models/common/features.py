""" Preprocessing and feature extraction, cached once per clip.

    python -m models.common.features --kind wav           # 16 kHz, silence trimmed waveforms
    python -m models.common.features --kind wavlm         # frozen WavLM Base+ hidden states
    python -m models.common.features --kind mel           # 80 band log mel spectrogram
    python -m models.common.features --kind bert --text_source gt    (or asr)

Speech preprocessing. Every clip is resampled to 16 kHz (TESS is recorded at 24.4 kHz and one clip at
96 kHz, and WavLM is trained on 16 kHz audio), the leading and trailing silence is trimmed (anything
30 dB below the clip's peak, padded back by 50 ms) and the clip is capped at 4 s. Clips are not
loudness normalised, because intensity is an emotional cue, and WavLM gets the zero mean, unit
variance input it was trained with.

Speech features (time steps x features). WavLM Base+ hidden states from the CNN output and all 12
transformer layers, [13, T, 768] at 50 frames per second, stored in float16. The speech model learns
how to weight the layers. The log mel baseline is [1, T, 80] at 100 frames per second.

Text preprocessing and features (tokens x features). Unicode normalisation and whitespace cleanup,
WordPiece tokenisation with the bert-base-uncased tokenizer (which lower cases), then the hidden
states of all 13 BERT layers, [13, N, 768] in float16, including [CLS] and [SEP].
"""

import argparse
import json
import os
import unicodedata

import numpy as np
import torch

from models.common.data import ROOT, load_manifest

CACHE = os.path.join(ROOT, "cache")
SR = 16000
MAX_S = 4.0
TOP_DB = 30
PAD_S = 0.05
WAVLM = "microsoft/wavlm-base-plus"
BERT = "bert-base-uncased"


def cache_dir(kind, text_source=None):
    return os.path.join(CACHE, kind if kind != "bert" else "bert_" + text_source)


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def preprocess_wav(path):
    import librosa
    import soundfile as sf
    audio, sr = sf.read(os.path.join(ROOT, path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    _, (start, end) = librosa.effects.trim(audio, top_db=TOP_DB, frame_length=400, hop_length=160)
    pad = int(PAD_S * SR)
    audio = audio[max(0, start - pad):min(len(audio), end + pad)][:int(MAX_S * SR)]
    return audio.astype(np.float32)


def extract_wav(clips):
    out = cache_dir("wav")
    os.makedirs(out, exist_ok=True)
    stats = {}
    for c in clips:
        a = preprocess_wav(c["path"])
        np.save(os.path.join(out, c["utt_id"] + ".npy"), a)
        stats[c["utt_id"]] = {"original_s": float(c["duration_s"]), "trimmed_s": round(len(a) / SR, 3)}
    with open(os.path.join(CACHE, "trim_stats.json"), "w") as f:
        json.dump(stats, f)
    d = np.array([s["trimmed_s"] for s in stats.values()])
    print("trimmed durations min {:.2f}, median {:.2f}, max {:.2f} s".format(d.min(), np.median(d), d.max()))


def load_wav(utt_id):
    return np.load(os.path.join(cache_dir("wav"), utt_id + ".npy"))


@torch.no_grad()
def extract_wavlm(clips):
    """One clip at a time. WavLM Base+ has group normalisation in its first convolution, which
    normalises over time, so zero padding in a batch would change the features of shorter clips."""
    from transformers import WavLMModel
    dev = device()
    model = WavLMModel.from_pretrained(WAVLM).to(dev).eval()
    out = cache_dir("wavlm")
    os.makedirs(out, exist_ok=True)
    for i, c in enumerate(clips):
        w = load_wav(c["utt_id"])
        w = (w - w.mean()) / (w.std() + 1e-7)
        hs = model(torch.from_numpy(w)[None].to(dev), output_hidden_states=True).hidden_states
        hs = torch.cat(hs, 0).float().cpu()                             # [13, T, 768]
        np.save(os.path.join(out, c["utt_id"] + ".npy"), hs.numpy().astype(np.float16))
        if i % 500 == 0:
            print("wavlm {}/{}".format(i + 1, len(clips)), flush=True)


def extract_mel(clips):
    import librosa
    out = cache_dir("mel")
    os.makedirs(out, exist_ok=True)
    for c in clips:
        m = librosa.feature.melspectrogram(y=load_wav(c["utt_id"]), sr=SR, n_fft=400, win_length=400,
                                           hop_length=160, n_mels=80, fmin=20, fmax=8000)
        logm = np.log(m + 1e-6).T.astype(np.float16)                    # [T, 80]
        np.save(os.path.join(out, c["utt_id"] + ".npy"), logm[None])


def clean_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def texts_for(clips, text_source):
    if text_source == "gt":
        return {c["utt_id"]: c["text"] for c in clips}
    with open(os.path.join(ROOT, "data", "asr_whisper.json")) as f:
        return json.load(f)


@torch.no_grad()
def extract_bert(clips, text_source, batch_size=64):
    from transformers import AutoModel, AutoTokenizer
    dev = device()
    tok = AutoTokenizer.from_pretrained(BERT)
    model = AutoModel.from_pretrained(BERT).to(dev).eval()
    texts = texts_for(clips, text_source)
    out = cache_dir("bert", text_source)
    os.makedirs(out, exist_ok=True)
    tokens = {}
    for i in range(0, len(clips), batch_size):
        batch = clips[i:i + batch_size]
        enc = tok([clean_text(texts[c["utt_id"]]) for c in batch], padding=True, return_tensors="pt")
        hs = model(**{k: v.to(dev) for k, v in enc.items()}, output_hidden_states=True).hidden_states
        hs = torch.stack(hs, 1).float().cpu()                           # [B, 13, N, 768]
        for j, c in enumerate(batch):
            n = int(enc["attention_mask"][j].sum())
            np.save(os.path.join(out, c["utt_id"] + ".npy"), hs[j, :, :n].numpy().astype(np.float16))
            tokens[c["utt_id"]] = tok.convert_ids_to_tokens(enc["input_ids"][j, :n])
    with open(os.path.join(out, "tokens.json"), "w") as f:
        json.dump(tokens, f)
    print("text_source {}, for example {}".format(text_source, tokens[clips[0]["utt_id"]]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["wav", "wavlm", "mel", "bert"])
    ap.add_argument("--text_source", default="gt", choices=["gt", "asr"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke tests)")
    args = ap.parse_args()
    clips = load_manifest()
    if args.limit:
        clips = clips[:args.limit]
    {"wav": extract_wav, "wavlm": extract_wavlm, "mel": extract_mel}.get(
        args.kind, lambda c: extract_bert(c, args.text_source))(clips)
    print("done", args.kind, len(clips), "clips ->", cache_dir(args.kind, args.text_source))
