""" Pack the code and the data and upload them to a private Hugging Face dataset repo, so a Colab
runtime can fetch them (see colab/*.ipynb). The same setup as Assignment 1.

    python scripts/colab_bundle.py --repo USER/ltrc-a2-data                 # code.zip only
    python scripts/colab_bundle.py --repo USER/ltrc-a2-data --data          # also the two datasets

code.zip         the repository without data, caches, checkpoints or results
A2a_data.zip     the provided clips (Assignment_2a/data/wav and transcripts)
TESS.zip         the 2,800 TESS clips, one copy (Assignment_2b/data/TESS/<speaker>_<emotion>/*.wav)
"""

import argparse
import os
import subprocess
import tempfile
import zipfile

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", "data", "cache", "checkpoints", "corpus", "mfa_tmp", "outputs",
             "Results", "Report", "downloads", ".ipynb_checkpoints"}
KEEP_DATA = ["Assignment_2b/data/manifest.csv", "Assignment_2b/data/splits.json"]


def zip_code(out):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for f in files:
                if f != ".DS_Store":
                    p = os.path.join(d, f)
                    z.write(p, os.path.relpath(p, ROOT))
        for rel in KEEP_DATA:
            if os.path.exists(os.path.join(ROOT, rel)):
                z.write(os.path.join(ROOT, rel), rel)
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        z.writestr("COMMIT", (commit or "no commit") + (" (with uncommitted changes)" if dirty else "") + "\n")


def zip_dir(src, arc_root, out):
    """Every file under src, stored as arc_root/<relative path>. Symlinked folders are followed."""
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for d, _, files in os.walk(src, followlinks=True):
            for f in sorted(files):
                if f.startswith("."):
                    continue
                p = os.path.join(d, f)
                z.write(p, os.path.join(arc_root, os.path.relpath(p, src)))
                n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        jobs = [("code.zip", zip_code)]
        if args.data:
            jobs += [("A2a_data.zip", lambda out: zip_dir(os.path.join(ROOT, "Assignment_2a", "data"), "data", out)),
                     ("TESS.zip", lambda out: zip_dir(os.path.join(ROOT, "Assignment_2b", "data", "TESS"), "TESS", out))]
        for name, fn in jobs:
            out = os.path.join(tmp, name)
            fn(out)
            with zipfile.ZipFile(out) as z:
                n = len(z.namelist())
            print("{} has {} files, {:.1f} MB, uploading".format(name, n, os.path.getsize(out) / 1e6), flush=True)
            api.upload_file(path_or_fileobj=out, path_in_repo=name, repo_id=args.repo, repo_type="dataset")
    print("done, https://huggingface.co/datasets/{}".format(args.repo))
