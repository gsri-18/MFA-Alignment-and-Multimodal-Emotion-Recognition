""" Training and testing shared by the three pipelines.

Each run is named <pipeline>_<variant>_<split>_seed<k>, for example speech_wavlm_word_seed0 or
fusion_wavlm-gt-gated_word_seed0. train() keeps the checkpoint with the best validation macro F1 in
checkpoints/<run>/best.pt and logs every epoch to Results/runs/<run>/train_log.csv. test() evaluates that
checkpoint on the test part of the split and writes metrics.json, predictions.csv, a confusion matrix and
embeddings.npz (the temporal, contextual and fusion vectors of every test clip, for the t-SNE plots).
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.common.data import EMOTIONS, ROOT, TessFeatures, collate, load_split
from models.common.nets import EmotionModel

CKPT = os.path.join(ROOT, "checkpoints")
RUNS = os.path.join(ROOT, "Results", "runs")


def parse_args(pipeline):
    ap = argparse.ArgumentParser(description="{} pipeline".format(pipeline))
    ap.add_argument("--split", default="word", choices=["word", "cross_speaker"])
    if pipeline in ("speech", "fusion"):
        ap.add_argument("--features", default="wavlm", choices=["wavlm", "mel"])
    if pipeline in ("text", "fusion"):
        ap.add_argument("--text_source", default="gt", choices=["gt", "asr"],
                        help="gt is the TESS transcript, asr is the Whisper transcript of the clip")
    if pipeline == "fusion":
        ap.add_argument("--fusion", default="gated", choices=["gated", "concat"])
        ap.add_argument("--modality_dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips of each part (smoke tests)")
    args = ap.parse_args()
    args.pipeline = pipeline
    return args


def run_name(args):
    if args.pipeline == "speech":
        variant = args.features
    elif args.pipeline == "text":
        variant = args.text_source
    else:
        variant = "{}-{}-{}".format(args.features, args.text_source, args.fusion)
        if args.modality_dropout:
            variant += "-md{}".format(args.modality_dropout)
    return "{}_{}_{}_seed{}".format(args.pipeline, variant, args.split, args.seed)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset(args, part):
    ids = load_split(args.split)[part]
    return TessFeatures(ids, speech=getattr(args, "features", None), text_source=getattr(args, "text_source", None),
                        limit=args.limit)


def loader(ds, args, shuffle):
    g = torch.Generator()
    g.manual_seed(args.seed)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate, generator=g,
                      num_workers=args.num_workers, persistent_workers=args.num_workers > 0)


def build_model(args, sample):
    shape = lambda x: (x.shape[0], x.shape[2])                          # (layers, dim)
    return EmotionModel(args.pipeline,
                        speech_shape=shape(sample["speech"]) if "speech" in sample else None,
                        text_shape=shape(sample["text"]) if "text" in sample else None,
                        fusion=getattr(args, "fusion", "gated"),
                        modality_dropout=getattr(args, "modality_dropout", 0.0))


def to(batch, dev):
    return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, dl, dev, keep=False):
    model.eval()
    losses, labels, probs, ids, reps = [], [], [], [], {}
    for batch in dl:
        batch = to(batch, dev)
        out = model(batch)
        losses.append(F.cross_entropy(out["logits"], batch["label"], reduction="sum").item())
        labels.append(batch["label"].cpu())
        probs.append(out["logits"].float().softmax(-1).cpu())
        ids += batch["utt_id"]
        if keep:
            for k in ("temporal", "contextual", "fusion", "gate"):
                if out.get(k) is not None:
                    reps.setdefault(k, []).append(out[k].float().cpu())
    labels, probs = torch.cat(labels).numpy(), torch.cat(probs).numpy()
    res = {"loss": sum(losses) / len(labels), **scores(labels, probs.argmax(1))}
    if keep:
        res.update(labels=labels, probs=probs, ids=ids, reps={k: torch.cat(v).numpy() for k, v in reps.items()})
    return res


def scores(y, p):
    from sklearn.metrics import accuracy_score, f1_score
    return {"accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=list(range(len(EMOTIONS))), zero_division=0))}


def train(pipeline):
    args = parse_args(pipeline)
    name = run_name(args)
    seed_everything(args.seed)
    dev = device()
    tr, va = dataset(args, "train"), dataset(args, "val")
    model = build_model(args, tr[0]).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print("{} with {} train and {} val clips, {:.2f} M trainable parameters, {}".format(
        name, len(tr), len(va), n_params / 1e6, dev), flush=True)
    dl_tr, dl_va = loader(tr, args, True), loader(va, args, False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    ckpt_dir, run_dir = os.path.join(CKPT, name), os.path.join(RUNS, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "train_log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1", "lr", "seconds"])
    best, best_epoch, t_start = -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for batch in dl_tr:
            batch = to(batch, dev)
            out = model(batch)
            loss = F.cross_entropy(out["logits"], batch["label"], label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(batch["label"])
            correct += (out["logits"].argmax(1) == batch["label"]).sum().item()
            n += len(batch["label"])
        v = evaluate(model, dl_va, dev)
        w.writerow([epoch, round(tot / n, 4), round(correct / n, 4), round(v["loss"], 4), round(v["accuracy"], 4),
                    round(v["macro_f1"], 4), "{:.2e}".format(sched.get_last_lr()[0]), round(time.time() - t0, 1)])
        log.flush()
        improved = v["macro_f1"] > best
        if improved:
            best, best_epoch = v["macro_f1"], epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val": v, "name": name}, os.path.join(ckpt_dir, "best.pt"))
        print("epoch {:2d}  train loss {:.3f} acc {:.3f} | val loss {:.3f} acc {:.3f} f1 {:.3f}{}".format(
            epoch, tot / n, correct / n, v["loss"], v["accuracy"], v["macro_f1"], "  *" if improved else ""),
            flush=True)
        if epoch - best_epoch >= args.patience:
            print("early stop, no val improvement for {} epochs".format(args.patience))
            break
    log.close()
    with open(os.path.join(run_dir, "train_summary.json"), "w") as f:
        json.dump({"run": name, "args": vars(args), "best_epoch": best_epoch, "best_val_macro_f1": best,
                   "parameters": n_params, "train_clips": len(tr), "val_clips": len(va),
                   "minutes": round((time.time() - t_start) / 60, 2), "layer_weights": model.layer_weights()},
                  f, indent=1)
    print("{} best val macro F1 {:.4f} at epoch {}".format(name, best, best_epoch))


def test(pipeline):
    from sklearn.metrics import classification_report, confusion_matrix
    args = parse_args(pipeline)
    name = run_name(args)
    dev = device()
    ckpt = torch.load(os.path.join(CKPT, name, "best.pt"), map_location="cpu", weights_only=False)
    te = dataset(args, "test")
    model = build_model(args, te[0])
    model.load_state_dict(ckpt["model"])
    model.to(dev)
    args.num_workers = min(args.num_workers, 2)
    r = evaluate(model, loader(te, args, False), dev, keep=True)
    y, p = r["labels"], r["probs"].argmax(1)
    report = classification_report(y, p, labels=list(range(len(EMOTIONS))), target_names=EMOTIONS,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=list(range(len(EMOTIONS))))
    run_dir = os.path.join(RUNS, name)
    os.makedirs(run_dir, exist_ok=True)
    metrics = {"run": name, "pipeline": pipeline, "split": args.split, "seed": args.seed,
               "variant": name[len(pipeline) + 1:name.rindex("_" + args.split)],
               "test_clips": len(y), "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
               "best_epoch": ckpt["epoch"], "val_macro_f1": ckpt["val"]["macro_f1"],
               "per_class": {e: {k: round(report[e][k], 4) for k in ("precision", "recall", "f1-score")}
                             for e in EMOTIONS},
               "confusion_matrix": cm.tolist(), "layer_weights": model.layer_weights()}
    if "gate" in r["reps"]:
        g = r["reps"]["gate"].mean(1)
        metrics["gate_mean"] = round(float(g.mean()), 4)
        metrics["gate_mean_per_emotion"] = {e: round(float(g[y == i].mean()), 4) for i, e in enumerate(EMOTIONS)
                                            if (y == i).any()}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    with open(os.path.join(run_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "true", "pred", "confidence"] + ["p_" + e for e in EMOTIONS])
        for u, t, pr in zip(r["ids"], y, r["probs"]):
            w.writerow([u, EMOTIONS[t], EMOTIONS[pr.argmax()], round(float(pr.max()), 4)] + [round(float(x), 4) for x in pr])
    np.savez_compressed(os.path.join(run_dir, "embeddings.npz"), ids=np.array(r["ids"]), labels=y,
                        **{k: v.astype(np.float16) for k, v in r["reps"].items() if k != "gate"})
    plot_confusion(cm, os.path.join(run_dir, "confusion_matrix.png"),
                   "{}\naccuracy {:.1%}, macro F1 {:.3f}".format(name, r["accuracy"], r["macro_f1"]))
    print("{} test accuracy {:.4f}, macro F1 {:.4f} ({} clips)".format(name, r["accuracy"], r["macro_f1"], len(y)))


def plot_confusion(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    short = ["angry", "disgust", "fear", "happy", "neutral", "surprise", "sad"]
    ax.set_xticks(range(len(short)), short, rotation=45, ha="right")
    ax.set_yticks(range(len(short)), short)
    for i in range(len(short)):
        for j in range(len(short)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
