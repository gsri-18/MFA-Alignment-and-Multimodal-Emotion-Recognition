""" Praat style alignment figures.

The spectrogram and pitch come from Praat itself through parselmouth (Praat's default settings, 5 ms
window, 0 to 5 kHz, 50 dB dynamic range), with the word and phone tiers drawn underneath like Praat's
TextGrid editor. The ISLE clips are short and drawn whole. The BU Radio News clips are about 25 to 30 s
long, so each gets an overview of the word tier and zoomed windows of 3 s.

    python plots.py      (or python align.py --stage plot)
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import parselmouth

from align import ANALYSIS, CORPUS, OUT, REFERENCE_RUN, RUNS, SILENCE, TG, read_textgrid, textgrids

FIG = os.path.join(OUT, "figures")
ZOOM_S = 3.0
RUN_COLOURS = {"arpa": "#1f5fbf", "g2p_en": "#d4732a", "mfa_ipa": "#2a9d5c", "arpa_adapted": "#b8336a"}


def wav_path(clip):
    return glob.glob(os.path.join(CORPUS, "*", clip + ".wav"))[0]


def draw_spectrogram(ax, snd, t0, t1):
    part = snd.extract_part(from_time=t0, to_time=t1, preserve_times=True)
    spec = part.to_spectrogram(window_length=0.005, maximum_frequency=5000)
    db = 10 * np.log10(np.maximum(spec.values, 1e-12))
    ax.pcolormesh(spec.x_grid(), spec.y_grid(), db, vmin=db.max() - 50, cmap="Greys", shading="flat",
                  rasterized=True)
    ax.set_ylim(0, 5000)
    ax.set_ylabel("Hz")
    pitch = part.to_pitch(pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    f0[f0 == 0] = np.nan
    ax2 = ax.twinx()
    ax2.plot(pitch.xs(), f0, ".", color="#1f5fbf", ms=2)
    ax2.set_ylim(50, 500)
    ax2.set_ylabel("F0 (Hz)", color="#1f5fbf")
    ax2.tick_params(axis="y", colors="#1f5fbf", labelsize=7)


def draw_tier(ax, intervals, t0, t1, label, colour="black", fontsize=8, min_label_s=0.0):
    """One TextGrid tier. Labels of intervals shorter than min_label_s are left out so they do not overlap."""
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    for s, e, text in intervals:
        if e < t0 or s > t1:
            continue
        for x in (s, e):
            if t0 <= x <= t1:
                ax.axvline(x, color=colour, lw=0.8)
        if text not in SILENCE and min(e, t1) - max(s, t0) >= min_label_s:
            ax.text((max(s, t0) + min(e, t1)) / 2, 0.5, text, ha="center", va="center", fontsize=fontsize,
                    color=colour, clip_on=True)
    ax.set_xlim(t0, t1)


def clip_figure(clip, run, t0, t1, out_path, title):
    snd = parselmouth.Sound(wav_path(clip))
    tiers = read_textgrid(textgrids(run)[clip])
    fig, axes = plt.subplots(4, 1, figsize=(max(8, 4 * (t1 - t0)), 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 2.6, 0.55, 0.55]})
    x = snd.xs()
    keep = (x >= t0) & (x <= t1)
    axes[0].plot(x[keep], snd.values[0][keep], color="black", lw=0.4)
    axes[0].set_yticks([])
    draw_spectrogram(axes[1], snd, t0, t1)
    for ax in axes[:2]:
        for s, e, w in tiers["words"]:
            if w not in SILENCE:
                for b in (s, e):
                    if t0 <= b <= t1:
                        ax.axvline(b, color="#1f5fbf", lw=0.6, alpha=0.6)
    draw_tier(axes[2], tiers["words"], t0, t1, "words", fontsize=9)
    draw_tier(axes[3], tiers["phones"], t0, t1, "phones", fontsize=7)
    axes[3].set_xlabel("time (s)")
    axes[0].set_title(title, fontsize=10)
    fig.tight_layout(h_pad=0.2)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def overview_figure(clip, run, out_path):
    """Whole clip with the word tier, for the long BU Radio News paragraphs."""
    snd = parselmouth.Sound(wav_path(clip))
    tiers = read_textgrid(textgrids(run)[clip])
    dur = snd.xmax
    fig, axes = plt.subplots(2, 1, figsize=(16, 3.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    axes[0].plot(snd.xs(), snd.values[0], color="black", lw=0.3)
    axes[0].set_yticks([])
    words = [iv for iv in tiers["words"] if iv[2] not in SILENCE]
    for i, (s, e, w) in enumerate(words):
        axes[0].axvspan(s, e, color="#1f5fbf" if i % 2 else "#d4732a", alpha=0.15, lw=0)
        axes[1].text((s + e) / 2, 0.5 + 0.3 * ((i % 3) - 1), w, ha="center", va="center", fontsize=6.5)
    axes[1].set_yticks([])
    axes[1].set_xlim(0, dur)
    axes[1].set_xlabel("time (s)")
    axes[0].set_title("{}  whole clip, word intervals shaded ({} run)".format(clip, run), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def comparison_figure(clip, t0, t1, out_path, runs=None, width=None):
    """Word and phone boundaries of every run (or of the runs given) over the same spectrogram window."""
    snd = parselmouth.Sound(wav_path(clip))
    runs = [r for r in (runs or RUNS) if clip in textgrids(r)]
    fig, axes = plt.subplots(1 + 2 * len(runs), 1, figsize=(width or max(8, 4 * (t1 - t0)), 2.2 + 1.0 * len(runs)),
                             sharex=True, gridspec_kw={"height_ratios": [2.4] + [0.45] * 2 * len(runs)})
    draw_spectrogram(axes[0], snd, t0, t1)
    for i, run in enumerate(runs):
        tiers = read_textgrid(textgrids(run)[clip])
        c = RUN_COLOURS.get(run, "black")
        for s, e, w in tiers["words"]:
            if w not in SILENCE:
                for b in (s, e):
                    if t0 <= b <= t1:
                        axes[0].axvline(b, color=c, lw=0.8, alpha=0.8)
        draw_tier(axes[1 + 2 * i], tiers["words"], t0, t1, run + " words", c, 8)
        draw_tier(axes[2 + 2 * i], tiers["phones"], t0, t1, run + " phones", c, 6.5)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title("{}  {:.2f} to {:.2f} s, boundaries of {}".format(
        clip, t0, t1, "every run" if len(runs) == len(RUNS) else " and ".join(runs)), fontsize=10)
    fig.tight_layout(h_pad=0.15)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def zoom_windows(clip, run, dur):
    """Short clips are drawn whole. Long clips get the first 3 s of speech and the 3 s around the
    phone that sits at the 30 ms floor most often in a cluster, which is where problems show up."""
    if dur <= 6:
        return [(0.0, dur)]
    with open(os.path.join(ANALYSIS, "summary.json")) as f:
        check = json.load(f)["checks"][run][clip]
    tiers = read_textgrid(textgrids(run)[clip])
    first = next(s for s, e, w in tiers["words"] if w not in SILENCE)
    windows = [(max(0.0, first - 0.2), min(dur, first - 0.2 + ZOOM_S))]
    floor = [s for s, _, _ in check["phones_at_floor"]]
    if floor:
        centre = max(floor, key=lambda t: sum(abs(t - u) < ZOOM_S / 2 for u in floor))
        t0 = min(max(0.0, centre - ZOOM_S / 2), dur - ZOOM_S)
        if abs(t0 - windows[0][0]) > ZOOM_S / 2:
            windows.append((t0, t0 + ZOOM_S))
    return windows


def plot_all():
    os.makedirs(FIG, exist_ok=True)
    run = REFERENCE_RUN
    for clip in textgrids(run):
        dur = parselmouth.Sound(wav_path(clip)).xmax
        if dur > 6:
            overview_figure(clip, run, os.path.join(FIG, "{}_overview.png".format(clip)))
        for k, (t0, t1) in enumerate(zoom_windows(clip, run, dur)):
            suffix = "" if dur <= 6 else "_zoom{}".format(k + 1)
            clip_figure(clip, run, t0, t1, os.path.join(FIG, "{}{}.png".format(clip, suffix)),
                        "{}  {:.2f} to {:.2f} s  ({} run, {} acoustic model, {} dictionary)".format(
                            clip, t0, t1, run, *RUNS[run]))
            comparison_figure(clip, t0, t1, os.path.join(FIG, "{}{}_runs.png".format(clip, suffix)))
    print(len(glob.glob(os.path.join(FIG, "*.png"))), "figures in", FIG)


if __name__ == "__main__":
    plot_all()
