""" The trainable blocks of the three pipelines.

speech   cached WavLM layers -> LayerMix -> projection -> 2 layer BiLSTM (temporal modelling)
         -> attentive pooling -> 256 d utterance vector -> classifier
text     cached BERT layers  -> LayerMix -> projection -> 2 layer Transformer encoder (contextual
         modelling) -> attentive pooling -> 256 d utterance vector -> classifier
fusion   both encoders above -> gated fusion (or concatenation) -> 256 d vector -> classifier

The pooled vectors are what the t-SNE plots show (temporal, contextual and fusion representations).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 7


class LayerMix(nn.Module):
    """Softmax weighted sum over the cached layers of a frozen encoder (as in SUPERB). Different
    layers carry different information (acoustic in the lower WavLM layers, more abstract higher up),
    and the task decides which to use. The learned weights are reported."""

    def __init__(self, n_layers):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x):                               # [B, L, T, D] -> [B, T, D]
        return (F.softmax(self.logits, 0)[None, :, None, None] * x).sum(1)

    def weights(self):
        return F.softmax(self.logits, 0).detach().cpu().tolist()


class AttentivePool(nn.Module):
    """Learned attention over time steps (or tokens). Emotion is often carried by a few frames,
    such as a stressed vowel, so a weighted mean works better than a plain mean."""

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h, mask):                         # [B, T, D], [B, T] -> [B, D], [B, T]
        s = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, -1)
        return (a.unsqueeze(-1) * h).sum(1), a


class SpeechEncoder(nn.Module):
    def __init__(self, n_layers, in_dim, d=256, dropout=0.2):
        super().__init__()
        self.mix = LayerMix(n_layers) if n_layers > 1 else None
        if in_dim == 80:
            # log mel baseline. Two 1-D convolutions learn local spectral patterns first.
            self.proj = nn.Sequential(
                nn.Conv1d(80, d, 5, padding=2), nn.BatchNorm1d(d), nn.GELU(),
                nn.Conv1d(d, d, 5, padding=2), nn.BatchNorm1d(d), nn.GELU())
            self.conv = True
        else:
            self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU())
            self.conv = False
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(d, d // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttentivePool(d)

    def forward(self, x, mask):
        x = self.mix(x) if self.mix is not None else x[:, 0]
        x = self.proj(x.transpose(1, 2)).transpose(1, 2) if self.conv else self.proj(x)
        x = self.drop(x)
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        return self.pool(h, mask)


class TextEncoder(nn.Module):
    def __init__(self, n_layers, in_dim=768, d=256, dropout=0.2):
        super().__init__()
        self.mix = LayerMix(n_layers) if n_layers > 1 else None
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.pool = AttentivePool(d)

    def forward(self, x, mask):
        x = self.mix(x) if self.mix is not None else x[:, 0]
        h = self.context(self.proj(x), src_key_padding_mask=~mask)
        return self.pool(h, mask)


class Classifier(nn.Module):
    def __init__(self, d=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, 128), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(128, N_CLASSES))

    def forward(self, z):
        return self.net(z)


class GatedFusion(nn.Module):
    """g = sigmoid(W [s, t]),  z = g * tanh(Ws s) + (1 - g) * tanh(Wt t).
    The gate is computed per dimension and per clip, so the model can decide how much to trust each
    modality. Its mean value (1 = speech only, 0 = text only) is reported."""

    def __init__(self, d=256):
        super().__init__()
        self.s, self.t = nn.Linear(d, d), nn.Linear(d, d)
        self.gate = nn.Linear(2 * d, d)

    def forward(self, s, t):
        g = torch.sigmoid(self.gate(torch.cat([s, t], -1)))
        return g * torch.tanh(self.s(s)) + (1 - g) * torch.tanh(self.t(t)), g


class ConcatFusion(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())

    def forward(self, s, t):
        return self.net(torch.cat([s, t], -1)), None


class EmotionModel(nn.Module):
    """One class for the three pipelines. pipeline is speech, text or fusion."""

    def __init__(self, pipeline, speech_shape=None, text_shape=None, fusion="gated", modality_dropout=0.0):
        super().__init__()
        self.pipeline = pipeline
        self.speech = SpeechEncoder(*speech_shape) if pipeline in ("speech", "fusion") else None
        self.text = TextEncoder(*text_shape) if pipeline in ("text", "fusion") else None
        self.fusion = (GatedFusion() if fusion == "gated" else ConcatFusion()) if pipeline == "fusion" else None
        self.modality_dropout = modality_dropout
        self.classifier = Classifier()

    def forward(self, batch):
        out = {}
        if self.speech is not None:
            out["temporal"], out["speech_attention"] = self.speech(batch["speech"], batch["speech_mask"])
        if self.text is not None:
            out["contextual"], out["text_attention"] = self.text(batch["text"], batch["text_mask"])
        if self.fusion is not None:
            s, t = out["temporal"], out["contextual"]
            if self.training and self.modality_dropout > 0:
                # randomly blank one modality for some clips, so neither branch is ignored outright
                drop = torch.rand(s.shape[0], device=s.device) < self.modality_dropout
                which = torch.rand(s.shape[0], device=s.device) < 0.5
                s = s.masked_fill((drop & which)[:, None], 0.0)
                t = t.masked_fill((drop & ~which)[:, None], 0.0)
            out["fusion"], out["gate"] = self.fusion(s, t)
            z = out["fusion"]
        else:
            z = out["temporal"] if self.speech is not None else out["contextual"]
        out["logits"] = self.classifier(z)
        return out

    def layer_weights(self):
        w = {}
        for name, enc in (("speech", self.speech), ("text", self.text)):
            if enc is not None and enc.mix is not None:
                w[name] = [round(x, 4) for x in enc.mix.weights()]
        return w
