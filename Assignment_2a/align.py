""" Forced alignment of the provided clips with the Montreal Forced Aligner.

The whole pipeline runs from this one script.

    python align.py                      # every stage
    python align.py --stage prepare      # or dictionaries | align | analyse | plot

prepare        converts the audio to 16 kHz RIFF WAV (three clips are NIST SPHERE files) and writes
               a normalised .lab transcript next to each clip, one folder per speaker
dictionaries   builds three pronunciation dictionaries for the corpus vocabulary
               arpa      english_us_arpa dictionary, missing words from the english_us_arpa G2P model
               g2p_en    CMUdict through g2p_en (every variant), its neural G2P for missing words, as in my earlier
                         FastSpeech 2 assignment
               mfa_ipa   english_mfa dictionary, missing words from the english_us_mfa G2P model
align          runs MFA once per configuration in RUNS, including an english_us_arpa model adapted to
               this corpus, and writes TextGrids to outputs/textgrids/<run>/
analyse        word and phone tables, alignment checks and the comparison between runs (outputs/analysis/)
plot           Praat style figures of every clip and the run comparison (outputs/figures/)

Needs the mfa command on PATH. Everything else runs in the regular Python environment.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
from collections import OrderedDict

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")              # the provided wav/ and transcripts/ folders
CORPUS = os.path.join(HERE, "corpus")          # MFA input, made by prepare
DICTS = os.path.join(HERE, "dictionaries")
OUT = os.path.join(HERE, "outputs")
TG = os.path.join(OUT, "textgrids")
ANALYSIS = os.path.join(OUT, "analysis")
MODELS = os.path.join(OUT, "models")
TMP = os.path.join(HERE, "mfa_tmp")
SR = 16000

# run name -> (acoustic model, dictionary). "adapted" is made by `mfa adapt` in the align stage.
RUNS = OrderedDict([
    ("arpa", ("english_us_arpa", "arpa")),
    ("g2p_en", ("english_us_arpa", "g2p_en")),
    ("mfa_ipa", ("english_mfa", "mfa_ipa")),
    ("arpa_adapted", ("adapted", "arpa")),
])
REFERENCE_RUN = "arpa"
# Pinned model versions. The v3.1.0 release of the english_mfa dictionary holds a reduced single
# pronunciation lexicon with Indian English phones (court is k ɒː ʈ), so the dictionary stays at v3.0.0,
# the multi dialect lexicon with pronunciation probabilities that matches the english_mfa acoustic model.
VERSIONS = {
    ("acoustic", "english_us_arpa"): "v3.0.0",
    ("acoustic", "english_mfa"): "v3.1.0",
    ("dictionary", "english_us_arpa"): "v3.0.0",
    ("dictionary", "english_mfa"): "v3.0.0",
    ("g2p", "english_us_arpa"): "v2.0.0a",
    ("g2p", "english_us_mfa"): "v3.0.0",
}
MFA_FLAGS = ["--clean", "--beam", "100", "--retry_beam", "400", "--output_format", "long_textgrid"]
SILENCE = {"", "sil", "sp", "spn", "<eps>"}


def mfa(*args):
    exe = shutil.which("mfa")
    assert exe, "Montreal Forced Aligner not found, see the README for the install command"
    cmd = [exe] + [str(a) for a in args]
    shown = ["***" if i and cmd[i - 1] == "--github_token" else c for i, c in enumerate(cmd)]
    print("$ " + " ".join(shown), flush=True)
    subprocess.run(cmd, check=True)


def mfa_root():
    return os.environ.get("MFA_ROOT_DIR", os.path.join(os.path.expanduser("~"), "Documents", "MFA"))


def download(kind, name):
    """Download a pretrained model at its pinned version unless it is already there.

    The file comes straight from the mfa-models release, since `mfa model download` first lists the
    releases through the GitHub API, which allows 60 anonymous calls per hour and is easily used up on
    a shared Colab address. If that fails it falls back to `mfa model download` (a GITHUB_TOKEN
    environment variable lifts the limit).
    """
    import urllib.request
    ext = ".dict" if kind == "dictionary" else ".zip"
    dst = os.path.join(mfa_root(), "pretrained_models", kind, name + ext)
    if os.path.exists(dst):
        return
    version = VERSIONS[(kind, name)]
    url = "https://github.com/MontrealCorpusTools/mfa-models/releases/download/{}-{}-{}/{}{}".format(
        kind, name, version, name, ext)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        print("downloading " + url, flush=True)
        urllib.request.urlretrieve(url, dst + ".part")
        os.replace(dst + ".part", dst)
        return
    except Exception as e:
        print("direct download failed ({}), trying mfa model download".format(e))
    args = ["model", "download", kind, name, "--version", version]
    if os.environ.get("GITHUB_TOKEN"):
        args += ["--github_token", os.environ["GITHUB_TOKEN"]]
    mfa(*args)


# ---------------------------------------------------------------- prepare

def speaker_of(stem, info):
    """BU Radio News clips carry the speaker in their SPHERE header (f2b). ISLE names hold the session."""
    m = re.search(r"speaker_id -s\d+ (\S+)", info)
    if m:
        return "bu_" + m.group(1).lower()
    m = re.search(r"(SESS\d+)", stem, re.I)
    return "isle_" + m.group(1).lower() if m else "spk_" + stem.lower()


def sphere_header(path):
    with open(path, "rb") as f:
        head = f.read(1024)
    return head.decode("latin-1") if head.startswith(b"NIST_1A") else ""


def normalise(text):
    """Transcript text to the word sequence MFA aligns.

    Dotted acronyms are spelled out (S.J.C.'s becomes s j c's), all capital words inside mixed case
    text are spelled out too (WBUR's becomes w b u r's, since the station name is read letter by
    letter), hyphens split words (nineteen seventy- six, de-politicize), and everything except
    letters and apostrophes is dropped. Numbers are already written as words in these transcripts.
    """
    text = " ".join(text.split())
    mixed_case = text != text.upper()

    def spell_dotted(m):
        letters = re.findall(r"[A-Za-z]", m.group(1))
        return " ".join(letters[:-1] + [letters[-1] + (m.group(2) or "")])
    text = re.sub(r"\b((?:[A-Za-z]\.){2,})('s)?", spell_dotted, text)

    if mixed_case:
        def spell_caps(m):
            word, suffix = m.group(1), m.group(2) or ""
            return " ".join(list(word[:-1]) + [word[-1] + suffix])
        text = re.sub(r"\b([A-Z]{2,})('s)?\b", spell_caps, text)

    text = text.replace("-", " ").lower()
    text = re.sub(r"[^a-z' ]", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


def prepare():
    wavs = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in glob.glob(os.path.join(DATA, "wav", "*"))}
    trans = sorted(glob.glob(os.path.join(DATA, "transcripts", "*")))
    assert trans, "put the provided wav/ and transcripts/ folders in " + DATA
    shutil.rmtree(CORPUS, ignore_errors=True)
    log = []
    for t in trans:
        stem = os.path.splitext(os.path.basename(t))[0]
        src = wavs[stem.lower()]
        audio, sr = sf.read(src, dtype="float32")
        assert sr == SR and audio.ndim == 1, (src, sr, audio.shape)
        header = sphere_header(src)
        spk = speaker_of(stem, header)
        os.makedirs(os.path.join(CORPUS, spk), exist_ok=True)
        sf.write(os.path.join(CORPUS, spk, stem + ".wav"), audio, SR, subtype="PCM_16")
        with open(t, encoding="utf-8", errors="replace") as f:
            original = f.read()
        words = normalise(original)
        with open(os.path.join(CORPUS, spk, stem + ".lab"), "w") as f:
            f.write(" ".join(words) + "\n")
        log.append({"file": stem, "speaker": spk, "source_format": "NIST SPHERE" if header else "RIFF WAV",
                    "duration_s": round(len(audio) / SR, 3), "original": " ".join(original.split()),
                    "normalised": " ".join(words), "n_words": len(words)})
        print("{:38s} {:15s} {:6.2f} s  {} words".format(stem, spk, len(audio) / SR, len(words)))
    os.makedirs(ANALYSIS, exist_ok=True)
    with open(os.path.join(ANALYSIS, "corpus.json"), "w") as f:
        json.dump(log, f, indent=1)


# ---------------------------------------------------------------- dictionaries

def corpus_words():
    words = set()
    for lab in glob.glob(os.path.join(CORPUS, "*", "*.lab")):
        with open(lab) as f:
            words.update(f.read().split())
    return sorted(words)


def pretrained_dict_path(name):
    download("dictionary", name)
    return os.path.join(mfa_root(), "pretrained_models", "dictionary", name + ".dict")


def read_dict(path):
    """word -> list of raw lines (MFA dictionaries may carry probability columns before the phones)."""
    entries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.setdefault(line.split()[0].lower(), []).append(line.rstrip("\n"))
    return entries


def dict_with_g2p(words, dict_name, g2p_name, out_path):
    """Entries of a pretrained dictionary for the corpus words, plus G2P pronunciations for the rest."""
    full = read_dict(pretrained_dict_path(dict_name))
    lines = [l for w in words for l in full.get(w, [])]
    oov = [w for w in words if w not in full]
    generated = []
    if oov:
        tmp = os.path.join(DICTS, "oov_" + dict_name)
        os.makedirs(tmp, exist_ok=True)
        with open(os.path.join(tmp, "words.txt"), "w") as f:
            f.write("\n".join(oov) + "\n")
        download("g2p", g2p_name)
        mfa("g2p", os.path.join(tmp, "words.txt"), g2p_name, os.path.join(tmp, "g2p.dict"), "--clean",
            "--num_pronunciations", "1", "--temporary_directory", os.path.join(TMP, "g2p_" + dict_name))
        with open(os.path.join(tmp, "g2p.dict")) as f:
            generated = [l.rstrip("\n") for l in f if l.strip()]
        shutil.rmtree(tmp)
    with open(out_path, "w") as f:
        f.write("\n".join(lines + generated) + "\n")
    return {"words": len(words), "in_pretrained_dictionary": len(words) - len(oov), "oov": oov,
            "g2p_pronunciations": generated, "entries": len(lines) + len(generated)}


def dict_g2p_en(words, out_path):
    """The lexicon builder from my earlier FastSpeech 2 assignment. CMUdict words get all their variants and MFA picks one per
    token, other words get the g2p_en neural prediction."""
    from g2p_en import G2p
    g2p = G2p()
    n, predicted = 0, {}
    with open(out_path, "w") as f:
        for w in words:
            prons = g2p.cmu.get(w)
            if not prons:
                prons = [g2p.predict(w)]
                predicted[w] = " ".join(prons[0])
            for p in prons:
                f.write("{}\t{}\n".format(w, " ".join(p)))
                n += 1
    return {"words": len(words), "in_cmudict": len(words) - len(predicted), "predicted": predicted, "entries": n}


def dictionaries():
    os.makedirs(DICTS, exist_ok=True)
    words = corpus_words()
    report = {
        "arpa": dict_with_g2p(words, "english_us_arpa", "english_us_arpa", os.path.join(DICTS, "arpa.dict")),
        "g2p_en": dict_g2p_en(words, os.path.join(DICTS, "g2p_en.dict")),
        "mfa_ipa": dict_with_g2p(words, "english_mfa", "english_us_mfa", os.path.join(DICTS, "mfa_ipa.dict")),
    }
    os.makedirs(ANALYSIS, exist_ok=True)
    with open(os.path.join(ANALYSIS, "dictionaries.json"), "w") as f:
        json.dump(report, f, indent=1)
    for name, r in report.items():
        print(name, {k: v for k, v in r.items() if k in ("words", "entries", "oov", "predicted")})


# ---------------------------------------------------------------- align

def align(num_jobs):
    os.makedirs(MODELS, exist_ok=True)
    for model in sorted({m for m, _ in RUNS.values() if m != "adapted"}):
        download("acoustic", model)
    adapted = os.path.join(MODELS, "english_us_arpa_adapted.zip")
    if any(m == "adapted" for m, _ in RUNS.values()):
        # MAP adaptation of the pretrained model to these two speakers and recording conditions
        mfa("adapt", CORPUS, os.path.join(DICTS, "arpa.dict"), "english_us_arpa", adapted, *MFA_FLAGS[:5],
            "--num_jobs", num_jobs, "--temporary_directory", os.path.join(TMP, "adapt"))
    for run, (model, dict_name) in RUNS.items():
        out = os.path.join(TG, run)
        shutil.rmtree(out, ignore_errors=True)
        mfa("align", CORPUS, os.path.join(DICTS, dict_name + ".dict"), adapted if model == "adapted" else model,
            out, *MFA_FLAGS, "--num_jobs", num_jobs, "--temporary_directory", os.path.join(TMP, run))
        n = len(glob.glob(os.path.join(out, "**", "*.TextGrid"), recursive=True))
        print("{} run, {} TextGrids".format(run, n))
        assert n == len(glob.glob(os.path.join(CORPUS, "*", "*.lab"))), "some clips failed to align in " + run


# ---------------------------------------------------------------- analyse

def read_textgrid(path):
    import tgt
    tg = tgt.io.read_textgrid(path, include_empty_intervals=True)
    tiers = {}
    for tier in tg.tiers:
        name = tier.name.split(" - ")[-1]      # MFA names tiers "<speaker> - words" for multi speaker corpora
        tiers[name] = [(round(i.start_time, 4), round(i.end_time, 4), i.text) for i in tier.intervals]
    return tiers


def textgrids(run):
    return {os.path.splitext(os.path.basename(p))[0]: p
            for p in sorted(glob.glob(os.path.join(TG, run, "**", "*.TextGrid"), recursive=True))}


def speech(intervals):
    return [iv for iv in intervals if iv[2] not in SILENCE]


def check_clip(tiers, expected_words, frame_floor=0.031):
    """Alignment checks for one clip.

    MFA phones usually last at least 3 frames of 10 ms, so a phone at that floor was squeezed into the
    shortest time the HMM allows, which often means the phone was not pronounced (a skipped phoneme).
    A very long speech phone usually absorbs a pause or a disfluency. Words that do not match the
    transcript, or spn phones, mean a word was not modelled at all.
    """
    words, phones = speech(tiers["words"]), speech(tiers["phones"])
    pauses = [iv for iv in tiers["words"] if iv[2] in SILENCE and iv[1] - iv[0] > 0]
    inner_pauses = [p for p in pauses if p[0] > words[0][0] and p[1] < words[-1][1]] if words else []
    phone_d = np.array([e - s for s, e, _ in phones])
    word_d = np.array([e - s for s, e, _ in words])
    return {
        "words_match_transcript": [w for _, _, w in words] == expected_words,
        "n_words": len(words), "n_phones": len(phones),
        "spn_phones": sum(p[2] == "spn" for p in tiers["phones"]),
        "leading_silence_s": round(words[0][0], 3) if words else None,
        "trailing_silence_s": round(tiers["words"][-1][1] - words[-1][1], 3) if words else None,
        "inner_pauses": len(inner_pauses), "inner_pause_s": round(sum(e - s for s, e, _ in inner_pauses), 3),
        "speech_rate_phones_per_s": round(len(phones) / max(1e-6, sum(word_d)), 2),
        "mean_phone_ms": round(1000 * phone_d.mean(), 1), "median_phone_ms": round(1000 * np.median(phone_d), 1),
        "phones_at_floor": [(s, e, p) for s, e, p in phones if e - s <= frame_floor],
        "phones_over_250ms": [(s, e, p) for s, e, p in phones if e - s > 0.25],
        "shortest_words": sorted(((round(e - s, 3), w, s) for s, e, w in words))[:3],
    }


def boundaries(tiers, tier):
    """Start and end of every speech interval, in order."""
    return np.array([(s, e) for s, e, _ in speech(tiers[tier])])


def compare(ref, other, tier):
    """Boundary differences in ms. Phones are compared only when both runs use the same phone set."""
    a, b = boundaries(ref, tier), boundaries(other, tier)
    if len(a) != len(b) or not len(a):
        return None
    if tier == "phones" and not {p for _, _, p in speech(ref[tier])} & {p for _, _, p in speech(other[tier])}:
        return None                         # ARPAbet and MFA IPA share no phone labels
    d = np.abs(a - b).ravel() * 1000
    return {"n_boundaries": int(d.size), "mean_ms": round(float(d.mean()), 1), "median_ms": round(float(np.median(d)), 1),
            "max_ms": round(float(d.max()), 1), "pct_over_20ms": round(100 * float((d > 20).mean()), 1),
            "pct_over_50ms": round(100 * float((d > 50).mean()), 1)}


def mfa_analysis_csv(run):
    """MFA writes alignment_analysis.csv (log likelihoods, phone duration deviation) next to the TextGrids."""
    path = os.path.join(TG, run, "alignment_analysis.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def analyse():
    os.makedirs(ANALYSIS, exist_ok=True)
    with open(os.path.join(ANALYSIS, "corpus.json")) as f:
        corpus = {c["file"]: c for c in json.load(f)}
    runs = [r for r in RUNS if os.path.isdir(os.path.join(TG, r))]
    all_tiers = {r: {k: read_textgrid(p) for k, p in textgrids(r).items()} for r in runs}

    # word and phone tables of every run
    for run in runs:
        for tier in ("words", "phones"):
            with open(os.path.join(ANALYSIS, "{}_{}.csv".format(run, tier)), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["file", "start", "end", "duration_ms", "label"])
                for clip, tiers in all_tiers[run].items():
                    for s, e, lab in tiers[tier]:
                        w.writerow([clip, s, e, round(1000 * (e - s), 1), lab])

    checks = {run: {clip: check_clip(t, corpus[clip]["normalised"].split()) for clip, t in all_tiers[run].items()}
              for run in runs}
    for run in runs:
        extra = mfa_analysis_csv(run)
        for clip, c in checks[run].items():
            row = extra.get(clip) or extra.get(clip + ".wav") or {}
            for key in ("overall_log_likelihood", "speech_log_likelihood", "phone_duration_deviation"):
                if key in row and row[key] not in ("", None):
                    c[key] = round(float(row[key]), 3)

    comparison = {}
    for run in runs:
        if run == REFERENCE_RUN:
            continue
        comparison[run] = {}
        for clip in all_tiers[REFERENCE_RUN]:
            comparison[run][clip] = {tier: compare(all_tiers[REFERENCE_RUN][clip], all_tiers[run][clip], tier)
                                     for tier in ("words", "phones")}
    with open(os.path.join(ANALYSIS, "summary.json"), "w") as f:
        json.dump({"runs": {r: {"acoustic_model": RUNS[r][0], "dictionary": RUNS[r][1]} for r in runs},
                   "checks": checks, "boundary_difference_vs_" + REFERENCE_RUN: comparison}, f, indent=1)
    write_markdown(runs, corpus, checks, comparison)
    print(open(os.path.join(ANALYSIS, "summary.md")).read())


def write_markdown(runs, corpus, checks, comparison):
    L = ["# Alignment summary", ""]
    L += ["| clip | speaker | duration (s) | words |", "|---|---|---|---|"]
    L += ["| {} | {} | {} | {} |".format(c, v["speaker"], v["duration_s"], v["n_words"]) for c, v in corpus.items()]
    L += ["", "## Checks per run", ""]
    L += ["| run | clip | words match | phones | 30 ms or less | over 250 ms | inner pauses (s) | phones/s | speech log likelihood |",
          "|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        for clip, c in checks[run].items():
            L.append("| {} | {} | {} | {} | {} | {} | {} ({}) | {} | {} |".format(
                run, clip, "yes" if c["words_match_transcript"] else "NO", c["n_phones"], len(c["phones_at_floor"]),
                len(c["phones_over_250ms"]), c["inner_pauses"], c["inner_pause_s"], c["speech_rate_phones_per_s"],
                c.get("speech_log_likelihood", "")))
    L += ["", "## Boundary difference from the {} run (ms)".format(REFERENCE_RUN), ""]
    L += ["| run | clip | word mean | word median | word max | word > 50 ms (%) | phone mean | phone > 20 ms (%) |",
          "|---|---|---|---|---|---|---|---|"]
    for run, clips in comparison.items():
        for clip, c in clips.items():
            w, p = c["words"], c["phones"]
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                run, clip, *(w[k] if w else "n/a" for k in ("mean_ms", "median_ms", "max_ms", "pct_over_50ms")),
                *(p[k] if p else "different phone set" for k in ("mean_ms", "pct_over_20ms"))))
    with open(os.path.join(ANALYSIS, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "prepare", "dictionaries", "align", "analyse", "plot"], default="all")
    ap.add_argument("--num_jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()
    stages = ["prepare", "dictionaries", "align", "analyse", "plot"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print("\n==== " + stage, flush=True)
        if stage == "prepare":
            prepare()
        elif stage == "dictionaries":
            dictionaries()
        elif stage == "align":
            align(args.num_jobs)
        elif stage == "analyse":
            analyse()
        elif stage == "plot":
            from plots import plot_all
            plot_all()
