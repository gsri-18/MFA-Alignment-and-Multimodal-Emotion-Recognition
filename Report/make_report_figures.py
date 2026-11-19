""" Figures sized for the two reports, redrawn from the committed outputs of the Colab runs.

    python Report/make_report_figures.py

The figures in Assignment_2a/outputs/figures and Assignment_2b/Results/plots are the ones the Colab runs made.
These versions show the same data at the size they are printed, with readable labels. Needs
Assignment_2a/corpus (python Assignment_2a/align.py --stage prepare) and Assignment_2b/cache/wav
(python -m models.common.features --kind wav, run in Assignment_2b) for the audio. The t-SNE maps are
recomputed from the saved embeddings, so their layout can differ slightly from the Colab plots, while the
silhouette scores, which use the embeddings themselves, do not change.
"""

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")
sys.path.insert(0, os.path.join(ROOT, "Assignment_2a"))
sys.path.insert(0, os.path.join(ROOT, "Assignment_2b"))

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7.5})
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
SHORT = ["angry", "disgust", "fear", "happy", "neutral", "surprise", "sad"]
COLOURS = ["#d62728", "#8c564b", "#9467bd", "#ff7f0e", "#7f7f7f", "#17becf", "#1f77b4"]
RUNS = os.path.join(ROOT, "Assignment_2b", "Results", "runs")


# ------------------------------------------------------------------ 2a

def alignment_bu():
    """F2BJRLP1 around 'important appointment governor', arpa run."""
    import parselmouth
    from align import read_textgrid, textgrids
    from plots import draw_spectrogram, draw_tier, wav_path
    clip, t0, t1 = "F2BJRLP1", 13.70, 14.76
    snd = parselmouth.Sound(wav_path(clip))
    tiers = read_textgrid(textgrids("arpa")[clip])
    fig, axes = plt.subplots(4, 1, figsize=(6.6, 3.9), sharex=True,
                             gridspec_kw={"height_ratios": [0.9, 2.0, 0.45, 0.45]})
    x = snd.xs()
    keep = (x >= t0) & (x <= t1)
    axes[0].plot(x[keep], snd.values[0][keep], color="black", lw=0.4)
    axes[0].set_yticks([])
    draw_spectrogram(axes[1], snd, t0, t1)
    draw_tier(axes[2], tiers["words"], t0, t1, "words", fontsize=8, min_label_s=0.05)
    draw_tier(axes[3], tiers["phones"], t0, t1, "phones", fontsize=6.3, min_label_s=0.025)
    axes[3].set_xlabel("time (s)")
    fig.tight_layout(h_pad=0.15)
    fig.savefig(bbox_inches="tight", pad_inches=0.03, fname=os.path.join(OUT, "a2a_bu_important.png"), dpi=220)
    plt.close(fig)


def alignment_isle():
    """'I said white not bait', arpa and mfa_ipa runs."""
    import parselmouth
    from align import read_textgrid, textgrids
    from plots import RUN_COLOURS, SILENCE, draw_spectrogram, draw_tier, wav_path
    clip, t0, t1 = "ISLE_SESS0131_BLOCKD02_01_sprt1", 0.0, 4.125
    snd = parselmouth.Sound(wav_path(clip))
    runs = ["arpa", "mfa_ipa"]
    fig, axes = plt.subplots(5, 1, figsize=(6.6, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.8, 0.42, 0.42, 0.42, 0.42]})
    draw_spectrogram(axes[0], snd, t0, t1)
    for i, run in enumerate(runs):
        tiers = read_textgrid(textgrids(run)[clip])
        c = RUN_COLOURS[run]
        for s, e, w in tiers["words"]:
            if w not in SILENCE:
                for b in (s, e):
                    axes[0].axvline(b, color=c, lw=0.8, alpha=0.8)
        draw_tier(axes[1 + 2 * i], tiers["words"], t0, t1, run + " words", c, 7.5)
        draw_tier(axes[2 + 2 * i], tiers["phones"], t0, t1, run + " phones", c, 6.0, min_label_s=0.09)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(h_pad=0.15)
    fig.savefig(bbox_inches="tight", pad_inches=0.03, fname=os.path.join(OUT, "a2a_isle_runs.png"), dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ 2b

def metrics(run):
    with open(os.path.join(RUNS, run, "metrics.json")) as f:
        return json.load(f)


def confusion():
    panels = [("speech_wavlm_cross_speaker_seed0", "Speech (WavLM)"),
              ("text_gt_cross_speaker_seed0", "Text (transcript)"),
              ("fusion_wavlm-gt-gated_cross_speaker_seed0", "Gated fusion")]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.55), sharey=True)
    for ax, (run, title) in zip(axes, panels):
        m = metrics(run)
        cm = np.array(m["confusion_matrix"])
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(7), SHORT, rotation=50, ha="right")
        ax.set_yticks(range(7), SHORT)
        for i in range(7):
            for j in range(7):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                            color="white" if norm[i, j] > 0.6 else "black")
        ax.set_title("{}, {:.1f}%".format(title, 100 * m["accuracy"]))
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("true")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(bbox_inches="tight", pad_inches=0.03, fname=os.path.join(OUT, "a2b_confusion.png"), dpi=220)
    plt.close(fig)


def speakers():
    with open(os.path.join(ROOT, "Assignment_2b", "data", "manifest.csv")) as f:
        return {r["utt_id"]: r["speaker"] for r in csv.DictReader(f)}


def tsne():
    from matplotlib.lines import Line2D
    from sklearn.manifold import TSNE
    spk = speakers()
    rows = [("word", "Unseen words"), ("cross_speaker", "Unseen speaker")]
    cols = [("speech_wavlm", "temporal", "Temporal (speech only)"),
            ("text_gt", "contextual", "Contextual (text only)"),
            ("fusion_wavlm-gt-gated", "fusion", "Fusion (gated)")]
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6))
    for r, (split, row_title) in enumerate(rows):
        for c, (prefix, key, col_title) in enumerate(cols):
            e = np.load(os.path.join(RUNS, "{}_{}_seed0".format(prefix, split), "embeddings.npz"))
            Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(e[key].astype(np.float32))
            s = np.array([spk[u] for u in e["ids"]])
            ax = axes[r, c]
            for i in range(7):
                for name, marker in (("OAF", "o"), ("YAF", "^")):
                    k = (e["labels"] == i) & (s == name)
                    if k.any():
                        ax.scatter(Z[k, 0], Z[k, 1], s=5, c=COLOURS[i], marker=marker, alpha=0.75, lw=0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_title)
            if c == 0:
                ax.set_ylabel(row_title)
    handles = [Line2D([], [], marker="o", ls="", color=COLOURS[i], label=SHORT[i], markersize=5) for i in range(7)]
    handles += [Line2D([], [], marker="o", ls="", color="black", label="OAF", markersize=4, fillstyle="none"),
                Line2D([], [], marker="^", ls="", color="black", label="YAF", markersize=4, fillstyle="none")]
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False, handletextpad=0.2, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.6)
    fig.savefig(bbox_inches="tight", pad_inches=0.03, fname=os.path.join(OUT, "a2b_tsne.png"), dpi=220)
    plt.close(fig)


def failure_cases():
    import librosa
    from models.common.features import SR, load_wav
    with open(os.path.join(RUNS, "fusion_wavlm-gt-gated_cross_speaker_seed0", "predictions.csv")) as f:
        preds = {r["utt_id"]: r for r in csv.DictReader(f)}
    cases = ["YAF_tire_angry", "YAF_yearn_pleasant_surprise", "YAF_bean_angry", "YAF_bath_fear", "YAF_jail_disgust"]
    fig, axes = plt.subplots(len(cases), 2, figsize=(6.6, 7.6), squeeze=False)
    for n, (row, utt) in enumerate(zip(axes, cases), 1):
        spk, word, true = utt.split("_", 2)
        pred = preds[utt]["pred"]
        for ax, emo in ((row[0], true), (row[1], pred)):
            y = load_wav("{}_{}_{}".format(spk, word, emo))
            m = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=400, hop_length=160, n_mels=80))
            ax.imshow(m, origin="lower", aspect="auto", extent=[0, len(y) / SR, 0, 80], cmap="magma")
            f0, _, _ = librosa.pyin(y, fmin=75, fmax=600, sr=SR, frame_length=1024, hop_length=160)
            ax2 = ax.twinx()
            ax2.plot(np.arange(len(f0)) * 160 / SR, f0, color="cyan", lw=1.1)
            ax2.set_ylim(50, 600)
            ax2.set_yticks([])
            ax.set_yticks([])
            ax.set_title("{}. “{}”, {}".format(n, word, emo.replace("_", " ")) if ax is row[0]
                         else "“{}”, {} (predicted)".format(word, emo.replace("_", " ")))
        if n == len(cases):
            row[0].set_xlabel("time (s)")
            row[1].set_xlabel("time (s)")
    fig.tight_layout(h_pad=1.2)
    fig.savefig(bbox_inches="tight", pad_inches=0.03, fname=os.path.join(OUT, "a2b_failure_cases.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.chdir(os.path.join(ROOT, "Assignment_2b"))     # the 2b modules resolve their data from here
    alignment_bu()
    alignment_isle()
    confusion()
    tsne()
    failure_cases()
    print("figures in", OUT)
