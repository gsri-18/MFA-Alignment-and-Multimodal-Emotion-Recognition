""" Writes the two Colab notebooks in colab/ (run once after editing the cells here).

    python scripts/make_notebooks.py

The notebooks committed in colab/ are the executed copies the Colab runs saved, with every output and cell
time. Running this script replaces them with blank ones.

Both notebooks are built to run unattended with Run all.
* The first cell starts a session log. Every cell prints and records the IST time it finished and how long it
  took (session_logs/cell_times.csv), and run() prefixes every output line with the time and keeps a copy
  (session_logs/run_output.txt).
* finish() saves the executed notebook with all its outputs, the logs and the result zips to the private
  Hugging Face dataset repo (results/<tag>/session_<start time>/) and to Google Drive
  (MyDrive/LTRC_Assignment2/<tag>/) when Drive is mounted, then disconnects and deletes the runtime so it stops
  using compute units.
* finish() runs after the last cell, after any cell that fails (not after a manual interrupt), and from a
  watchdog when the notebook runs longer than MAX_HOURS.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_REPO = "gsri-18/ltrc-a2-data"

SESSION = '''TAG = "__TAG__"
DATA_REPO = "__REPO__"
MAX_HOURS = __HOURS__   # if still running after this, save everything and disconnect anyway

import datetime, json, os, shutil, threading, time

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
LOG_DIR = "/content/session_logs"
os.makedirs(LOG_DIR, exist_ok=True)
DRIVE_DIR = None        # MyDrive/LTRC_Assignment2/<TAG>, set below when Google Drive is mounted
ARTIFACTS = []          # result zips, copied to Drive as they are made


def now():
    return datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{now()} IST] {msg}"
    print(line, flush=True)
    with open(f"{LOG_DIR}/session_log.txt", "a") as f:
        f.write(line + "\\n")


def save_notebook():
    """The notebook as it is at this moment, with every output so far."""
    try:
        from google.colab import _message
        nb = _message.blocking_request("get_ipynb", request="", timeout_sec=120)["ipynb"]
        with open(f"{LOG_DIR}/{TAG}_executed.ipynb", "w") as f:
            json.dump(nb, f)
        log(f"exported the notebook with its outputs ({len(nb['cells'])} cells)")
    except Exception as e:
        log(f"could not export the notebook ({type(e).__name__} {e})")


def to_drive(path):
    if DRIVE_DIR and os.path.exists(path):
        try:
            if os.path.isdir(path):
                shutil.copytree(path, f"{DRIVE_DIR}/{os.path.basename(path)}_{STAMP}", dirs_exist_ok=True)
            else:
                shutil.copy(path, DRIVE_DIR)
        except Exception as e:
            log(f"Drive copy of {path} failed ({type(e).__name__} {e})")


def save_everything(status):
    with open(f"{LOG_DIR}/status.json", "w") as f:
        json.dump({"notebook": TAG, "status": status, "started": f"{START:%Y-%m-%d %H:%M:%S} IST",
                   "ended": f"{now()} IST", "minutes": round((time.time() - START.timestamp()) / 60, 1)}, f, indent=1)
    save_notebook()
    try:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=LOG_DIR, path_in_repo=f"results/{TAG}/session_{STAMP}", repo_id=DATA_REPO,
                              repo_type="dataset", commit_message=f"{TAG} session {STAMP} {status}")
        log(f"saved the logs and the notebook to {DATA_REPO}/results/{TAG}/session_{STAMP}")
    except Exception as e:
        log(f"Hugging Face upload of the logs failed ({type(e).__name__} {e})")
    if DRIVE_DIR:
        for p in ARTIFACTS:
            to_drive(p)
        to_drive(LOG_DIR)
        try:
            from google.colab import drive
            drive.flush_and_unmount()
            log(f"saved everything to Google Drive {DRIVE_DIR}")
        except Exception as e:
            log(f"Drive flush failed ({type(e).__name__} {e})")


_done = threading.Lock()


def finish(status="completed"):
    """Save everything to Hugging Face and Drive, then disconnect and delete the runtime."""
    if not _done.acquire(blocking=False):
        return
    log(f"notebook {status} after {(time.time() - START.timestamp()) / 60:.1f} min, saving")
    save_everything(status)
    log("disconnecting and deleting the runtime")
    time.sleep(3)
    from google.colab import runtime
    runtime.unassign()


if "_session_hooks" not in globals():
    START = datetime.datetime.now(IST)
    STAMP = f"{START:%Y%m%d_%H%M%S}"

    def _cell_start(info):
        globals()["_cell_t0"] = time.time()
        globals()["_cell_first_line"] = (info.raw_cell.strip().splitlines() or [""])[0][:90]

    def _cell_end(result):
        took = time.time() - globals().get("_cell_t0", time.time())
        err = result.error_before_exec or result.error_in_exec
        status = "ok" if err is None else type(err).__name__
        print(f"[{now()} IST] finished in {took:.1f} s" + ("" if err is None else f", {status}"))
        path = f"{LOG_DIR}/cell_times.csv"
        new = not os.path.exists(path)
        with open(path, "a") as f:
            if new:
                f.write("finished_ist,seconds,status,first_line\\n")
            f.write('{},{:.1f},{},"{}"\\n'.format(now(), took, status, globals().get("_cell_first_line", "session setup").replace('"', "'")))
        if err is not None and not isinstance(err, KeyboardInterrupt):
            log(f"cell failed with {err!r}")
            finish("failed")

    get_ipython().events.register("pre_run_cell", _cell_start)
    get_ipython().events.register("post_run_cell", _cell_end)
    _watchdog = threading.Timer(MAX_HOURS * 3600, finish, args=(f"stopped by the {MAX_HOURS} h limit",))
    _watchdog.daemon = True
    _watchdog.start()
    _session_hooks = True
    log(f"session started, {TAG}, everything is saved and the runtime deleted at the end, after an error "
        f"or after {MAX_HOURS} h")'''

TOKEN = '''import getpass, requests
from huggingface_hub import whoami


def can_write(token):
    """Asks the data repo how a small file would be uploaded, which needs write access and commits nothing."""
    r = requests.post(f"https://huggingface.co/api/datasets/{DATA_REPO}/preupload/main", timeout=30,
                      headers={"authorization": f"Bearer {token}"},
                      json={"files": [{"path": "results/write_check.txt", "size": 4, "sample": "dGVzdA=="}]})
    return r.status_code == 200


token = os.environ.get("HF_TOKEN", "")
if not token:
    try:
        from google.colab import userdata          # a Colab secret named HF_TOKEN, if there is one
        token = userdata.get("HF_TOKEN") or ""
    except Exception:
        token = ""
if not token:
    token = getpass.getpass("Hugging Face token with write access ").strip()
if not can_write(token):                           # checked now, not after an hour of training
    raise PermissionError(f"this Hugging Face token has no write access to {DATA_REPO}, use a token with write "
                          "access (and update or delete the HF_TOKEN Colab secret if there is one)")
os.environ["HF_TOKEN"] = token
log(f"logged in to Hugging Face as {whoami()['name']}, with write access to {DATA_REPO}")

# Google Drive gets a second copy of everything. Declining the prompt keeps Hugging Face only.
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = f"/content/drive/MyDrive/LTRC_Assignment2/{TAG}"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    log(f"Google Drive mounted, copies go to MyDrive/LTRC_Assignment2/{TAG}")
except Exception as e:
    log(f"Google Drive not mounted ({type(e).__name__}), saving to Hugging Face only")'''

RUN = '''import re, subprocess, sys

PY = sys.executable  # Colab's Python, not the one in the aligner's environment


def run(cmd, keep=None, drop=None):
    """Run a shell command, print its filtered output with the time of each line, keep a copy in
    session_logs/run_output.txt and raise if it fails."""
    log(f"$ {cmd}")
    with open(f"{LOG_DIR}/run_output.txt", "a") as out:
        out.write(f"\\n[{now()} IST] $ {cmd}\\n")
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            for part in line.replace("\\r", "\\n").split("\\n"):
                if part.strip() and (not keep or re.search(keep, part)) and not (drop and re.search(drop, part)):
                    stamped = f"[{datetime.datetime.now(IST):%H:%M:%S}] {part}"
                    print(stamped, flush=True)
                    out.write(stamped + "\\n")
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{cmd} failed with exit code {code}")'''

FETCH = '''import zipfile
from huggingface_hub import hf_hub_download

for name, dst in __FILES__:
    path = hf_hub_download(DATA_REPO, name, repo_type="dataset", force_download=(name == "code.zip"))
    zipfile.ZipFile(path).extractall(dst)
    log(f"{name} -> {dst}")
log("code version " + open("/content/project/COMMIT").read().strip())'''

FINISH_MD = '''## Save and disconnect

Saves the executed notebook, the logs, the cell timings and the results to Hugging Face and Google Drive, then
disconnects and deletes this runtime.'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def session(tag, hours):
    return SESSION.replace("__TAG__", tag).replace("__REPO__", DATA_REPO).replace("__HOURS__", str(hours))


def notebook(cells, gpu):
    meta = {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "A100"
    return {"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": cells}


UNATTENDED = '''
**Unattended run.** Choose Run all, then answer the two prompts at the start (the Hugging Face token, or a Colab
secret named `HF_TOKEN`, and the Google Drive permission). After that nothing needs watching. Every cell and every
log line carries its IST time. When the last cell finishes, or a cell fails, or the run passes __HOURS__ h, the
notebook saves itself with all outputs, the logs and the results to Hugging Face (`__REPO__`,
`results/__TAG__/`) and Google Drive (`MyDrive/LTRC_Assignment2/__TAG__/`), and then disconnects and deletes the
runtime. Interrupting a cell by hand does not disconnect.
'''


def unattended(tag, hours):
    return UNATTENDED.replace("__TAG__", tag).replace("__REPO__", DATA_REPO).replace("__HOURS__", str(hours))


A2A_HOURS, A2B_HOURS = 1.5, 6

A2A = [
    md('''
# Assignment 2a, forced alignment with the Montreal Forced Aligner

Runs `Assignment_2a/align.py` on a Colab CPU runtime (alignment needs no GPU). Each stage stops the notebook if it
fails.

1. **prepare** converts the six clips to 16 kHz WAV (three are NIST SPHERE files) and normalises the transcripts
2. **dictionaries** builds three lexicons for the 151 words (english_us_arpa with G2P for missing words, g2p_en as in
   Assignment 1, english_mfa with G2P for missing words)
3. **align** runs MFA four times (english_us_arpa with two lexicons, english_mfa, and english_us_arpa adapted to
   these clips)
4. **analyse** writes word and phone tables, alignment checks and the boundary differences between runs
5. **plot** draws Praat style figures

The code and the provided clips come from the private dataset repo `gsri-18/ltrc-a2-data`, and the outputs go
back to it as `results/A2a_outputs.zip`.
''' + unattended("A2a", A2A_HOURS)),
    code(session("A2a", A2A_HOURS)),
    code(TOKEN),
    md("## Setup"),
    code(FETCH.replace("__FILES__", '[("code.zip", "/content/project"), ("A2a_data.zip", "/content/project/Assignment_2a")]') + '''
!ls /content/project/Assignment_2a/data/*
!pip -q install "setuptools<81" tgt praat-parselmouth g2p_en

''' + RUN),
    md("Montreal Forced Aligner gets its own micromamba environment, and only its `mfa` command is used."),
    code('''
%cd /content
run("curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba")
run("./bin/micromamba create -q -y -p /content/mfa_env -c conda-forge python=3.11 montreal-forced-aligner=3",
    drop=r".")
os.environ["PATH"] = "/content/mfa_env/bin:" + os.environ["PATH"]
os.environ["MFA_ROOT_DIR"] = "/content/mfa_root"
run("mfa version")
'''),
    md("## Prepare"),
    code('''
%cd /content/project/Assignment_2a
run(f"{PY} align.py --stage prepare")
!head -c 300 corpus/*/*.lab
'''),
    md("## Dictionaries"),
    code('''
run(f"{PY} align.py --stage dictionaries", drop=r"it/s|━")
!wc -l dictionaries/*.dict
'''),
    md("## Align"),
    code('''
run(f"{PY} align.py --stage align --num_jobs {os.cpu_count()}", keep=r"^\\$|INFO|TextGrids|rror|Traceback|downloading")
!find outputs/textgrids -name "*.TextGrid" | sort
'''),
    md("## Analyse"),
    code('''
run(f"{PY} align.py --stage analyse")
'''),
    md("## Plot"),
    code('''
run(f"{PY} align.py --stage plot")
from IPython.display import Image, display
for f in ["ISLE_SESS0131_BLOCKD02_03_sprt1_runs.png", "F2BJRLP1_zoom1.png", "F2BJRLP1_overview.png"]:
    print(f)
    display(Image(f"outputs/figures/{f}"))
'''),
    md("## Upload"),
    code('''
from huggingface_hub import HfApi

%cd /content/project/Assignment_2a
with zipfile.ZipFile("/content/A2a_outputs.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for top in ["outputs", "dictionaries"]:
        for d, _, files in os.walk(top):
            for f in files:
                if not f.endswith(".zip"):          # the adapted acoustic model stays out (about 50 MB)
                    z.write(os.path.join(d, f))
    for d, _, files in os.walk("corpus"):
        for f in files:
            if f.endswith(".lab"):
                z.write(os.path.join(d, f))
log(f"A2a_outputs.zip {os.path.getsize('/content/A2a_outputs.zip') / 1e6:.1f} MB")
ARTIFACTS.append("/content/A2a_outputs.zip")
to_drive("/content/A2a_outputs.zip")                # Drive first, so a failed upload loses nothing
try:
    HfApi().upload_file(path_or_fileobj="/content/A2a_outputs.zip", path_in_repo="results/A2a_outputs.zip",
                        repo_id=DATA_REPO, repo_type="dataset")
    log("uploaded results/A2a_outputs.zip")
except Exception as e:
    log(f"Hugging Face upload failed ({type(e).__name__} {e}), the zip is in Google Drive")
'''),
    md(FINISH_MD),
    code('finish("completed")'),
]

UPLOAD_2B = '''from huggingface_hub import HfApi


def upload_results():
    """Results/, the checkpoints and the Whisper transcripts, as results/A2b_results.zip in the data repo and in Drive."""
    out = "/content/A2b_results.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for top in ["Results", "checkpoints"]:
            for d, _, files in os.walk(f"/content/project/Assignment_2b/{top}"):
                for f in files:
                    p = os.path.join(d, f)
                    z.write(p, os.path.relpath(p, "/content/project/Assignment_2b"))
        asr = "/content/project/Assignment_2b/data/asr_whisper.json"
        if os.path.exists(asr):
            z.write(asr, "data/asr_whisper.json")
    if out not in ARTIFACTS:
        ARTIFACTS.append(out)
    to_drive(out)                                   # Drive first, so a failed upload loses nothing
    try:
        HfApi().upload_file(path_or_fileobj=out, path_in_repo="results/A2b_results.zip", repo_id=DATA_REPO,
                            repo_type="dataset", commit_message=f"A2b results {now()} IST")
        log(f"uploaded results/A2b_results.zip ({os.path.getsize(out) / 1e6:.1f} MB)"
            + (", copied to Drive" if DRIVE_DIR else ""))
    except Exception as e:
        log(f"Hugging Face upload failed ({type(e).__name__} {e})"
            + (", the zip is in Google Drive" if DRIVE_DIR else ", the run continues"))


# a previous session's results, so finished runs are skipped
try:
    zipfile.ZipFile(hf_hub_download(DATA_REPO, "results/A2b_results.zip", repo_type="dataset",
                                    force_download=True)).extractall("/content/project/Assignment_2b")
    log(f"restored {len(os.listdir('/content/project/Assignment_2b/Results/runs'))} finished runs")
except Exception as e:
    log(f"no earlier results ({type(e).__name__})")'''

A2B = [
    md('''
# Assignment 2b, multimodal emotion recognition on TESS

Trains and tests the speech only, text only and multimodal pipelines on a Colab GPU (an A100 if available, any GPU
works).

The code and the 2,800 TESS clips come from the private dataset repo `gsri-18/ltrc-a2-data`. Results and
checkpoints are uploaded to it as `results/A2b_results.zip` after each split, and running all cells again after a
lost session restores them and skips the finished runs.

1. **data** checks the 2,800 clips and writes the manifest and the two splits (word disjoint, cross speaker)
2. **features** caches the trimmed 16 kHz audio, log mel spectrograms, WavLM Base+ and BERT hidden states, and
   Whisper transcripts with their BERT features
3. **experiments** trains and tests 7 variants on 2 splits with 3 seeds (42 runs)
4. **analysis** writes the accuracy tables, t-SNE plots, confusion matrices and failure cases
''' + unattended("A2b", A2B_HOURS)),
    code(session("A2b", A2B_HOURS)),
    code(TOKEN),
    md("## Setup"),
    code('''
import sys, torch
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
log(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {os.cpu_count()} CPU cores")
'''),
    code(FETCH.replace("__FILES__", '[("code.zip", "/content/project"), ("TESS.zip", "/content/project/Assignment_2b/data")]') + '''
import transformers, librosa, sklearn
log(f"transformers {transformers.__version__}, librosa {librosa.__version__}, scikit-learn {sklearn.__version__}")

''' + RUN + '''


''' + UPLOAD_2B),
    md("## Data"),
    code('''
%cd /content/project/Assignment_2b
run(f"{PY} scripts/prepare_tess.py")
'''),
    md('''
## Features

Trimmed 16 kHz audio and log mel spectrograms, then the frozen encoders (WavLM Base+ for speech, BERT for text).
'''),
    code('''
for kind in ["wav", "mel", "wavlm"]:
    run(f"{PY} -W ignore -m models.common.features --kind {kind}", drop=r"(?i)warn|unexpected|ignored")
run(f"{PY} -W ignore -m models.common.features --kind bert --text_source gt", drop=r"(?i)warn|unexpected|ignored|^- ")
!du -sh cache/*
'''),
    md('''
Whisper transcripts of every clip, the second text source. The TESS transcript is the same carrier phrase for all
seven emotions, so this tests whether text that comes from speech recognition carries any emotional cue.
'''),
    code('''
run(f"{PY} -W ignore scripts/transcribe_whisper.py", drop=r"(?i)warn|deprecat")
run(f"{PY} -W ignore -m models.common.features --kind bert --text_source asr", drop=r"(?i)warn|unexpected|ignored|^- ")
'''),
    md('''
## Experiments

Word disjoint split first (each word is only in train, val or test), then the cross speaker split (train on OAF,
test on YAF). Each run prints its size, best validation epoch and test result.
'''),
    code('''
run(f"{PY} scripts/run_experiments.py --split word --seeds 0 1 2")
upload_results()
'''),
    code('''
run(f"{PY} scripts/run_experiments.py --split cross_speaker --seeds 0 1 2")
upload_results()
'''),
    md("## Analysis"),
    code('''
run(f"{PY} -W ignore scripts/analyse.py", drop=r"(?i)warn")
from IPython.display import Image, Markdown, display
for f in ["accuracy_tables.md", "per_emotion.md", "fusion_vs_speech.md", "failure_cases.md"]:
    display(Markdown(open(f"Results/{f}").read()))
for f in sorted(os.listdir("Results/plots")):
    print(f)
    display(Image(f"Results/plots/{f}"))
upload_results()
'''),
    md(FINISH_MD),
    code('finish("completed")'),
]


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "colab"), exist_ok=True)
    for name, cells, gpu in (("Assignment_2a_alignment.ipynb", A2A, False), ("Assignment_2b_emotion.ipynb", A2B, True)):
        with open(os.path.join(ROOT, "colab", name), "w") as f:
            json.dump(notebook(cells, gpu), f, indent=1)
        print("wrote colab/" + name)
