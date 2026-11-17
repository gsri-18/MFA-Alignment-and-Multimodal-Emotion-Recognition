""" Whisper transcripts of every TESS clip, the second text source (--text_source asr).

    python scripts/transcribe_whisper.py

The TESS transcript is the same carrier phrase for every emotion, so it cannot tell emotions apart.
A deployed system would get its text from speech recognition instead, and Whisper's transcript can
carry traces of how a sentence was said (an exclamation or question mark, a misheard word). This
tests whether that makes the text branch useful. Writes data/asr_whisper.json ({utt_id: text}) and
reports the word error rate against the TESS transcripts.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.common.data import ROOT, load_manifest
from models.common.features import SR, load_wav

MODEL = "openai/whisper-large-v3-turbo"


def words(t):
    return re.sub(r"[^a-z' ]", " ", t.lower()).split()


def wer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)], len(ref)


if __name__ == "__main__":
    import argparse
    from transformers import pipeline
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke tests)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    asr = pipeline("automatic-speech-recognition", model=args.model, device=dev,
                   dtype=torch.float16 if dev == "cuda" else torch.float32)
    clips = load_manifest()[:args.limit or None]
    audio = [{"raw": load_wav(c["utt_id"]), "sampling_rate": SR} for c in clips]
    texts = asr(audio, batch_size=32, generate_kwargs={"language": "english", "task": "transcribe"})
    out = {c["utt_id"]: t["text"].strip() for c, t in zip(clips, texts)}
    if not args.limit:
        with open(os.path.join(ROOT, "data", "asr_whisper.json"), "w") as f:
            json.dump(out, f, indent=0)
    errs = [wer(words(c["text"]), words(out[c["utt_id"]])) for c in clips]
    print("word error rate {:.2%} over {} clips".format(sum(e for e, _ in errs) / sum(n for _, n in errs), len(clips)))
    for c in clips[:5]:
        print(c["utt_id"], "->", out[c["utt_id"]])
