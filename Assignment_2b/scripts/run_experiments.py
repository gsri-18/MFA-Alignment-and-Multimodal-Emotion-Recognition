""" Trains and tests every model variant of the comparison, for one split or both.

    python scripts/run_experiments.py --split word --seeds 0 1 2

A run whose Results/runs/<run>/metrics.json already exists is skipped, so after a lost Colab session the
same command continues where it stopped.

Variants
speech   WavLM features (main) and log mel features (baseline)
text     TESS transcript (main) and Whisper transcript
fusion   WavLM + TESS transcript, gated (main), WavLM + Whisper transcript, gated, and
         WavLM + TESS transcript with plain concatenation instead of the gate
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = [
    ("speech", ["--features", "wavlm"]),
    ("speech", ["--features", "mel"]),
    ("text", ["--text_source", "gt"]),
    ("text", ["--text_source", "asr"]),
    ("fusion", ["--features", "wavlm", "--text_source", "gt", "--fusion", "gated"]),
    ("fusion", ["--features", "wavlm", "--text_source", "asr", "--fusion", "gated"]),
    ("fusion", ["--features", "wavlm", "--text_source", "gt", "--fusion", "concat"]),
]


def name(pipeline, extra, split, seed):
    opts = dict(zip(extra[::2], extra[1::2]))
    if pipeline == "speech":
        variant = opts["--features"]
    elif pipeline == "text":
        variant = opts["--text_source"]
    else:
        variant = "{}-{}-{}".format(opts["--features"], opts["--text_source"], opts["--fusion"])
    return "{}_{}_{}_seed{}".format(pipeline, variant, split, seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["word", "cross_speaker"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--only", default="", help="run only variants whose run name contains this text")
    args, passthrough = ap.parse_known_args()
    t_all = time.time()
    for split in args.split:
        for seed in args.seeds:
            for pipeline, extra in GRID:
                run = name(pipeline, extra, split, seed)
                if args.only and args.only not in run:
                    continue
                if os.path.exists(os.path.join(ROOT, "Results", "runs", run, "metrics.json")):
                    print("skip", run, "(already tested)")
                    continue
                t0 = time.time()
                common = extra + ["--split", split, "--seed", str(seed)] + passthrough
                for stage in ("train", "test"):
                    script = os.path.join(ROOT, "models", pipeline + "_pipeline", stage + ".py")
                    out = subprocess.run([sys.executable, "-W", "ignore", script] + common, cwd=ROOT,
                                         capture_output=True, text=True)
                    if out.returncode:
                        print(out.stdout[-3000:], out.stderr[-3000:])
                        raise SystemExit("{} {} failed".format(run, stage))
                    lines = out.stdout.strip().splitlines()
                    if stage == "train":
                        print(lines[0])
                        print("  " + lines[-1])
                    else:
                        print("  " + lines[-1])
                print("  {:.1f} min".format((time.time() - t0) / 60), flush=True)
    print("all done in {:.1f} min".format((time.time() - t_all) / 60))
