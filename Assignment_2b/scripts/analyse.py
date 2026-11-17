""" Tables, plots and the error analysis from the test results of every run.

    python scripts/analyse.py

Reads Results/runs/*/ and writes
Results/accuracy_tables.md (and .csv)   accuracy and macro F1 of every variant, mean and std over seeds
Results/per_emotion.md                  recall of each emotion for the main models
Results/fusion_vs_speech.md             clip by clip, where fusion fixes or breaks the speech prediction
Results/failure_cases.md                misclassified clips with what each pipeline predicted
Results/plots/                          t-SNE of the temporal, contextual and fusion representations,
                                        confusion matrices, training curves, layer weights, spectrograms
"""

import csv
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.common.data import EMOTIONS, ROOT, load_manifest

RES = os.path.join(ROOT, "Results")
RUNS = os.path.join(RES, "runs")
PLOTS = os.path.join(RES, "plots")
SPLITS = ["word", "cross_speaker"]
MAIN = {"speech": "speech_wavlm", "text": "text_gt", "fusion": "fusion_wavlm-gt-gated"}
ASR = {"speech": "speech_wavlm", "text": "text_asr", "fusion": "fusion_wavlm-asr-gated"}
COLOURS = ["#d62728", "#8c564b", "#9467bd", "#ff7f0e", "#7f7f7f", "#17becf", "#1f77b4"]
SHORT = {"pleasant_surprise": "surprise"}


def load_runs():
    runs = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "metrics.json"))):
        with open(p) as f:
            m = json.load(f)
        runs[m["run"]] = m
    return runs


def run_id(prefix, split, seed):
    return "{}_{}_seed{}".format(prefix, split, seed)


def predictions(run):
    path = os.path.join(RUNS, run, "predictions.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {r["utt_id"]: r for r in csv.DictReader(f)}


def pm(xs):
    return "{:.1f} ± {:.1f}".format(100 * np.mean(xs), 100 * np.std(xs)) if len(xs) > 1 else "{:.1f}".format(100 * xs[0])


# ------------------------------------------------------------------ tables

def accuracy_tables(runs):
    groups = defaultdict(list)
    for m in runs.values():
        groups[(m["split"], m["pipeline"], m["variant"])].append(m)
    order = {"speech": 0, "text": 1, "fusion": 2}
    rows = sorted(groups.items(), key=lambda kv: (SPLITS.index(kv[0][0]), order[kv[0][1]], kv[0][2]))
    L = ["# Test accuracy of every model variant", "",
         "Mean ± standard deviation over seeds, in percent. Chance is 14.3 % (7 balanced classes).", ""]
    csv_rows = []
    for split in SPLITS:
        sub = [(k, v) for k, v in rows if k[0] == split]
        if not sub:
            continue
        n_test = sub[0][1][0]["test_clips"]
        L += ["## Split `{}` ({} test clips)".format(split, n_test), "",
              "| pipeline | variant | accuracy | macro F1 | seeds |", "|---|---|---|---|---|"]
        for (s, pipe, variant), ms in sub:
            acc, f1 = [m["accuracy"] for m in ms], [m["macro_f1"] for m in ms]
            L.append("| {} | {} | {} | {} | {} |".format(pipe, variant, pm(acc), pm(f1), len(ms)))
            csv_rows.append([split, pipe, variant, len(ms), round(100 * np.mean(acc), 2), round(100 * np.std(acc), 2),
                             round(100 * np.mean(f1), 2), round(100 * np.std(f1), 2)])
        L.append("")
    with open(os.path.join(RES, "accuracy_tables.md"), "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(RES, "accuracy_tables.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "pipeline", "variant", "seeds", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"])
        w.writerows(csv_rows)
    print("\n".join(L))


def per_emotion(runs):
    L = ["# Recall of each emotion (test, mean over seeds, percent)", ""]
    for split in SPLITS:
        cols = [(k, p) for k, p in list(MAIN.items()) + [("text asr", ASR["text"]), ("fusion asr", ASR["fusion"])]
                if any(r.startswith(p + "_" + split) for r in runs)]
        if not cols:
            continue
        L += ["## Split `{}`".format(split), "", "| emotion | " + " | ".join(k for k, _ in cols) + " |",
              "|---|" + "---|" * len(cols)]
        for e in EMOTIONS:
            cells = []
            for _, prefix in cols:
                rec = [m["per_class"][e]["recall"] for r, m in runs.items() if r.startswith(prefix + "_" + split + "_")]
                cells.append("{:.1f}".format(100 * np.mean(rec)) if rec else "")
            L.append("| {} | {} |".format(e, " | ".join(cells)))
        L.append("")
    with open(os.path.join(RES, "per_emotion.md"), "w") as f:
        f.write("\n".join(L))


def fusion_vs_speech(runs):
    """Clip by clip, the speech errors that fusion fixes and the correct speech predictions it breaks."""
    L = ["# Where fusion changes the speech only prediction", "",
         "Same split and seed, clip by clip. fixed = speech wrong and fusion right, broken = speech right and "
         "fusion wrong.", "", "| split | fusion variant | seed | speech acc | fusion acc | fixed | broken | gate mean |",
         "|---|---|---|---|---|---|---|---|"]
    for split in SPLITS:
        for fusion in sorted({m["variant"] for m in runs.values() if m["pipeline"] == "fusion"}):
            feat = fusion.split("-")[0]
            for seed in range(10):
                f_run, s_run = run_id("fusion_" + fusion, split, seed), run_id("speech_" + feat, split, seed)
                fp, sp = predictions(f_run), predictions(s_run)
                if not fp or not sp:
                    continue
                fixed = sum(sp[u]["pred"] != sp[u]["true"] and fp[u]["pred"] == fp[u]["true"] for u in fp)
                broken = sum(sp[u]["pred"] == sp[u]["true"] and fp[u]["pred"] != fp[u]["true"] for u in fp)
                L.append("| {} | {} | {} | {:.1f} | {:.1f} | {} | {} | {} |".format(
                    split, fusion, seed, 100 * runs[s_run]["accuracy"], 100 * runs[f_run]["accuracy"], fixed, broken,
                    runs[f_run].get("gate_mean", "")))
    with open(os.path.join(RES, "fusion_vs_speech.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------ plots

def tsne_panel(ax, X, y, spk, title):
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X.astype(np.float32))
    sil = silhouette_score(X.astype(np.float32), y) if len(set(y)) > 1 else float("nan")
    for i, e in enumerate(EMOTIONS):
        for s, marker in (("OAF", "o"), ("YAF", "^")):
            k = (y == i) & (spk == s)
            if k.any():
                ax.scatter(Z[k, 0], Z[k, 1], s=9, c=COLOURS[i], marker=marker, alpha=0.75, lw=0,
                           label=SHORT.get(e, e) if s == "OAF" else None)
    ax.set_title("{}\nsilhouette {:.2f} (in the {}-d space)".format(title, sil, X.shape[1]), fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return sil


def embeddings(run):
    path = os.path.join(RUNS, run, "embeddings.npz")
    return np.load(path) if os.path.exists(path) else None


def tsne_figures():
    speakers = {c["utt_id"]: c["speaker"] for c in load_manifest()}
    sils = {}
    for split in SPLITS:
        for tag, cfg in (("gt", MAIN), ("asr", ASR)):
            panels = [(cfg["speech"], "temporal", "speech only, temporal block (BiLSTM)"),
                      (cfg["text"], "contextual", "text only ({}), contextual block (Transformer)".format(tag)),
                      (cfg["fusion"], "fusion", "multimodal ({}), fusion block (gated)".format(tag))]
            panels = [(run_id(p, split, 0), k, t) for p, k, t in panels if embeddings(run_id(p, split, 0)) is not None]
            if len(panels) < 2:
                continue
            fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8))
            for ax, (run, key, title) in zip(np.atleast_1d(axes), panels):
                e = embeddings(run)
                spk = np.array([speakers[u] for u in e["ids"]])
                sils["{} {} {}".format(split, tag, key)] = round(float(tsne_panel(ax, e[key], e["labels"], spk, title)), 4)
            np.atleast_1d(axes)[0].legend(fontsize=7, loc="best", markerscale=1.5)
            fig.suptitle("t-SNE of the test clips, split {}, seed 0 (circles OAF, triangles YAF)".format(split), fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS, "tsne_{}_{}.png".format(split, tag)), dpi=150)
            plt.close(fig)
        # the three blocks inside the one fusion model
        e = embeddings(run_id(MAIN["fusion"], split, 0))
        if e is not None:
            spk = np.array([speakers[u] for u in e["ids"]])
            fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
            for ax, key in zip(axes, ("temporal", "contextual", "fusion")):
                sils["{} fusion-model {}".format(split, key)] = round(float(
                    tsne_panel(ax, e[key], e["labels"], spk, "inside the fusion model, {} block".format(key))), 4)
            axes[0].legend(fontsize=7, loc="best", markerscale=1.5)
            fig.suptitle("t-SNE of the three blocks of {}".format(run_id(MAIN["fusion"], split, 0)), fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS, "tsne_{}_inside_fusion.png".format(split)), dpi=150)
            plt.close(fig)
    with open(os.path.join(RES, "silhouette_scores.json"), "w") as f:
        json.dump(sils, f, indent=1)
    print("silhouette", sils)


def confusion_grid(runs):
    for split in SPLITS:
        items = [(k, run_id(p, split, 0)) for k, p in MAIN.items() if run_id(p, split, 0) in runs]
        if not items:
            continue
        fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 4.6))
        for ax, (k, r) in zip(np.atleast_1d(axes), items):
            cm = np.array(runs[r]["confusion_matrix"])
            norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            names = [SHORT.get(e, e) for e in EMOTIONS]
            ax.set_xticks(range(7), names, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(7), names, fontsize=8)
            for i in range(7):
                for j in range(7):
                    if cm[i, j]:
                        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                                color="white" if norm[i, j] > 0.6 else "black")
            ax.set_title("{} ({})\naccuracy {:.1%}".format(k, r, runs[r]["accuracy"]), fontsize=9)
            ax.set_xlabel("predicted")
        np.atleast_1d(axes)[0].set_ylabel("true")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, "confusion_{}.png".format(split)), dpi=150)
        plt.close(fig)


def training_curves():
    for split in SPLITS:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        found = False
        for k, p in list(MAIN.items()) + [("text asr", ASR["text"]), ("fusion asr", ASR["fusion"])]:
            path = os.path.join(RUNS, run_id(p, split, 0), "train_log.csv")
            if not os.path.exists(path):
                continue
            found = True
            with open(path) as f:
                rows = list(csv.DictReader(f))
            ep = [int(r["epoch"]) for r in rows]
            axes[0].plot(ep, [float(r["train_loss"]) for r in rows], label=k + " train")
            axes[0].plot(ep, [float(r["val_loss"]) for r in rows], "--", label=k + " val")
            axes[1].plot(ep, [float(r["val_accuracy"]) for r in rows], label=k)
        if not found:
            plt.close(fig)
            continue
        axes[0].set_ylabel("cross entropy")
        axes[1].set_ylabel("validation accuracy")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.legend(fontsize=7)
        fig.suptitle("training curves, split {}, seed 0".format(split), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, "training_curves_{}.png".format(split)), dpi=150)
        plt.close(fig)


def layer_weights(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, (key, prefix, title) in zip(axes, (("speech", MAIN["speech"], "WavLM layer weights (speech only)"),
                                               ("text", MAIN["text"], "BERT layer weights (text only)"))):
        for split in SPLITS:
            ws = [m["layer_weights"].get(key) for r, m in runs.items() if r.startswith(prefix + "_" + split + "_")]
            ws = [w for w in ws if w]
            if ws:
                ax.plot(range(len(ws[0])), np.mean(ws, 0), "o-", label=split)
        ax.axhline(1 / 13, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("layer (0 = embedding output)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "layer_weights.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ failure cases

def failure_cases(runs, n=5):
    from models.common.features import SR, load_wav
    import librosa
    clips = {c["utt_id"]: c for c in load_manifest()}
    asr_path = os.path.join(ROOT, "data", "asr_whisper.json")
    asr = json.load(open(asr_path)) if os.path.exists(asr_path) else {}
    picked = []
    for split in SPLITS:
        fp = predictions(run_id(MAIN["fusion"], split, 0))
        if not fp:
            continue
        others = {k: predictions(run_id(p, split, 0)) or {} for k, p in
                  (("speech", MAIN["speech"]), ("text", MAIN["text"]), ("fusion asr", ASR["fusion"]))}
        errors = sorted((r for r in fp.values() if r["pred"] != r["true"]), key=lambda r: -float(r["confidence"]))
        seen_pairs = set()
        for r in errors:                                   # most confident errors, one per confusion pair
            pair = (r["true"], r["pred"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            picked.append((split, r, {k: v.get(r["utt_id"], {}).get("pred", "") for k, v in others.items()}))
            if len(picked) >= n:
                break
        if len(picked) >= n:
            break
    L = ["# Failure cases", "",
         "The most confident errors of the multimodal model ({}, seed 0), one per confusion pair, from the word "
         "split first and then the cross speaker split.".format(MAIN["fusion"]), "",
         "| # | split | clip | true | fusion predicts (confidence) | speech only | text only | fusion with ASR text | Whisper transcript |",
         "|---|---|---|---|---|---|---|---|---|"]
    for i, (split, r, o) in enumerate(picked, 1):
        L.append("| {} | {} | {} | {} | {} ({:.2f}) | {} | {} | {} | {} |".format(
            i, split, r["utt_id"], r["true"], r["pred"], float(r["confidence"]), o["speech"], o["text"],
            o["fusion asr"], asr.get(r["utt_id"], "")))
    with open(os.path.join(RES, "failure_cases.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    if not picked:
        return
    fig, axes = plt.subplots(len(picked), 2, figsize=(11, 2.3 * len(picked)), squeeze=False)
    for row, (split, r, _) in zip(axes, picked):
        c = clips[r["utt_id"]]
        # the clip next to the same word and speaker said with the emotion the model predicted
        for ax, emo, tag in ((row[0], c["emotion"], "true " + c["emotion"]),
                             (row[1], r["pred"], "same word, " + r["pred"] + " (the predicted class)")):
            u = "{}_{}_{}".format(c["speaker"], c["word"], emo)
            y = load_wav(u)
            m = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=400, hop_length=160, n_mels=80))
            ax.imshow(m, origin="lower", aspect="auto", extent=[0, len(y) / SR, 0, 80], cmap="magma")
            f0, _, _ = librosa.pyin(y, fmin=75, fmax=600, sr=SR, frame_length=1024, hop_length=160)
            ax2 = ax.twinx()
            ax2.plot(np.arange(len(f0)) * 160 / SR, f0, color="cyan", lw=1.2)
            ax2.set_ylim(50, 600)
            ax2.set_yticks([])
            ax.set_title("{}  ({})".format(u, tag), fontsize=8)
            ax.set_yticks([])
    fig.suptitle("failure cases, log mel spectrogram with F0 (cyan, 50 to 600 Hz)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "failure_cases.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(PLOTS, exist_ok=True)
    runs = load_runs()
    print(len(runs), "runs with test results")
    accuracy_tables(runs)
    per_emotion(runs)
    fusion_vs_speech(runs)
    confusion_grid(runs)
    training_curves()
    layer_weights(runs)
    tsne_figures()
    failure_cases(runs)
