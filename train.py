"""DL Project 1 (Kaggle): a ResNet under 5M trainable parameters for CIFAR-10. Trains, validates, writes a submission CSV."""
import argparse
import copy
import csv
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_kaggle = sorted(glob.glob("/kaggle/input/**/train_labels.csv", recursive=True))
DATA = os.path.dirname(_kaggle[0]) if _kaggle else os.path.join(HERE, "data.noindex")  # Kaggle notebook or laptop
RUNS = "/kaggle/working/runs" if os.path.isdir("/kaggle/working") else os.path.join(HERE, "runs")
CACHE = "/kaggle/working" if os.path.isdir("/kaggle/working") else DATA                # Kaggle input is read-only
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------- data ----------------
def load(split):
    """Decode the PNGs once, then reuse a uint8 cache: x (N,32,32,3), y (N,), ids (N,)."""
    cache = os.path.join(CACHE, f"{split}.npz")
    if not os.path.exists(cache):
        from PIL import Image
        rows = list(csv.DictReader(open(os.path.join(DATA, "train_labels.csv" if split == "train" else "test.csv"))))
        x = np.stack([np.asarray(Image.open(os.path.join(DATA, f"{split}_images", r["filename"])).convert("RGB")) for r in rows])
        np.savez(cache, x=x, y=np.array([int(r.get("label", -1)) for r in rows]), ids=np.array([int(r["id"]) for r in rows]))
    z = np.load(cache)
    return z["x"], z["y"], z["ids"]


def augment(x, pad=4, cutout=16):
    """Random crop (reflect pad), horizontal flip and cutout, done on the GPU for a whole batch."""
    B, C, H, W = x.shape
    d = x.device
    if pad:
        xp = F.pad(x, (pad,) * 4, mode="reflect")
        iy = torch.randint(0, 2 * pad + 1, (B, 1), device=d) + torch.arange(H, device=d)
        ix = torch.randint(0, 2 * pad + 1, (B, 1), device=d) + torch.arange(W, device=d)
        x = xp[torch.arange(B, device=d)[:, None, None, None], torch.arange(C, device=d)[None, :, None, None],
               iy[:, None, :, None], ix[:, None, None, :]]
    x = torch.where(torch.rand(B, 1, 1, 1, device=d) < 0.5, x.flip(3), x)
    if cutout:
        cy = torch.randint(0, H, (B, 1, 1), device=d)
        cx = torch.randint(0, W, (B, 1, 1), device=d)
        ys = torch.arange(H, device=d)[None, :, None]
        xs = torch.arange(W, device=d)[None, None, :]
        hole = ((ys - cy).abs() < cutout // 2) & ((xs - cx).abs() < cutout // 2)
        x = x * (~hole)[:, None].to(x.dtype)
    return x


def trivial_aug(x):
    """TrivialAugment-style: each image gets ONE random op at a random strength. x is a [0,1] batch on the GPU.
    Ops: identity, brightness, contrast, saturation, sharpness, posterize, solarize, autocontrast,
    rotate, shear x/y, translate x/y. Every op is computed for the whole batch, then each image keeps its own."""
    B, d = x.shape[0], x.device
    op = torch.randint(0, 13, (B,), device=d)
    m = torch.rand(B, device=d)                                                   # strength in [0, 1]
    sgn = torch.where(torch.rand(B, device=d) < 0.5, -1.0, 1.0)
    f = (1 + 0.9 * m * sgn).view(B, 1, 1, 1)                                      # blend factor in [0.1, 1.9]
    v = lambda t: t.view(B, 1, 1, 1)
    gray = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:])
    blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"),
                    torch.tensor([[1., 1, 1], [1, 5, 1], [1, 1, 1]], device=d).div(13).expand(3, 1, 3, 3), groups=3)
    q = v(2 ** torch.floor(m * 5))                                                # posterize: drop 0..4 bits
    lo, hi = x.amin((2, 3), keepdim=True), x.amax((2, 3), keepdim=True)
    cand = [x, x * f, v(gray.mean((1, 2, 3))) + f * (x - v(gray.mean((1, 2, 3)))), gray + f * (x - gray),
            blur + f * (x - blur), torch.floor((x * 255 + 1e-3) / q) * q / 255, torch.where(x >= v(1 - m), 1 - x, x),
            (x - lo) / (hi - lo).clamp_min(1e-3)]
    out = torch.zeros_like(x)
    for k, c in enumerate(cand):
        out = torch.where(v(op == k), c, out)
    # geometric ops: one affine warp per image (identity for the colour ops)
    theta = torch.zeros(B, 2, 3, device=d)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    s = sgn * m
    ang = s * math.radians(30)
    r = op == 8
    theta[r, 0, 0], theta[r, 0, 1], theta[r, 1, 0], theta[r, 1, 1] = ang[r].cos(), -ang[r].sin(), ang[r].sin(), ang[r].cos()
    theta[op == 9, 0, 1] = 0.3 * s[op == 9]                                       # shear x
    theta[op == 10, 1, 0] = 0.3 * s[op == 10]                                     # shear y
    theta[op == 11, 0, 2] = 0.5 * s[op == 11]                                     # translate x, up to 8 px
    theta[op == 12, 1, 2] = 0.5 * s[op == 12]                                     # translate y, up to 8 px
    geo = op >= 8
    if geo.any():
        grid = F.affine_grid(theta[geo], list(x[geo].shape), align_corners=False)
        out[geo] = F.grid_sample(x[geo], grid, padding_mode="zeros", align_corners=False)
    return out.clamp(0, 1)


def zoom(x, lo=0.75, hi=1.33):
    """Half of the images get a random scale. Above 1 the object shrinks, below 1 it grows. x is a [0,1] batch."""
    B, d = x.shape[0], x.device
    s = torch.empty(B, device=d).uniform_(lo, hi)
    s = torch.where(torch.rand(B, device=d) < 0.5, s, torch.ones_like(s))
    th = torch.zeros(B, 2, 3, device=d)
    th[:, 0, 0] = th[:, 1, 1] = s
    grid = F.affine_grid(th, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False).clamp(0, 1)


def lowpass(x, p=0.5):
    """With probability p, remove fine detail with a Gaussian low-pass filter of random strength. x is a [0,1] batch."""
    B, d = x.shape[0], x.device
    f = torch.fft.fftfreq(x.shape[-1], d=1 / x.shape[-1], device=d)
    r2 = f[None] ** 2 + f[:, None] ** 2
    s = torch.empty(B, device=d).uniform_(4, 16).view(B, 1, 1, 1)             # frequency sigma, in cycles per image
    y = torch.fft.ifft2(torch.fft.fft2(x) * torch.exp(-r2[None, None] / (2 * s ** 2))).real.clamp(0, 1)
    return torch.where(torch.rand(B, 1, 1, 1, device=d) < p, y, x)


def augmix(x, width=3):
    """AugMix-style (Hendrycks 2020): blend several TrivialAugment copies of each image with the original."""
    B, d = x.shape[0], x.device
    w = torch.from_numpy(np.random.dirichlet(np.ones(width), B)).float().to(d)
    m = torch.rand(B, device=d).view(B, 1, 1, 1)
    mix = sum(w[:, i].view(B, 1, 1, 1) * trivial_aug(x) for i in range(width))
    return (1 - m) * x + m * mix


# ---------------- model ----------------
ACTS = {"relu": lambda: nn.ReLU(inplace=True), "gelu": nn.GELU, "silu": lambda: nn.SiLU(inplace=True),
        "mish": lambda: nn.Mish(inplace=True)}


class SE(nn.Module):
    """Squeeze-and-excitation (Hu 2018): rescale each channel from a global summary of the block output."""
    def __init__(self, c, r, act):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // r, 1), act(), nn.Conv2d(c // r, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)


class Blur(nn.Module):
    """Anti-aliased downsampling (Zhang 2019): a fixed [1,2,1] low-pass filter, then every second pixel."""
    def __init__(self, c):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        self.register_buffer("k", ((k[None] * k[:, None]) / 16).expand(c, 1, 3, 3).clone())
        self.c = c

    def forward(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), self.k, stride=2, groups=self.c)


class ResBlock(nn.Module):
    """Shared forward pass: y = act(S(x) + SE(F(x))), with optional stochastic depth on F."""
    def forward(self, x):
        y = self.se(self.f(x))
        if self.training and self.sd > 0:                                     # drop the branch per image
            y = y * ((torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.sd).to(y.dtype) / (1 - self.sd))
        return self.act(self.s(x) + y)

    def shortcut(self, ci, co, stride, blur):
        if stride == 1 and ci == co:
            return nn.Identity()
        if blur and stride > 1:                                               # ResNet-D shortcut: pool, then project
            return nn.Sequential(nn.AvgPool2d(stride), nn.Conv2d(ci, co, 1, bias=False), nn.BatchNorm2d(co))
        return nn.Sequential(nn.Conv2d(ci, co, 1, stride, bias=False), nn.BatchNorm2d(co))


class BasicBlock(ResBlock):
    """y = act(S(x) + F(x)). F = conv-BN-act-conv-BN. S = identity, or a 1x1 conv-BN projection when the shape changes."""
    def __init__(self, ci, co, stride, act="relu", se=0, blur=False, sd=0.0):
        super().__init__()
        A = ACTS[act]
        down = blur and stride > 1
        layers = [nn.Conv2d(ci, co, 3, 1 if down else stride, 1, bias=False), nn.BatchNorm2d(co), A()]
        layers += [Blur(co)] if down else []
        self.f = nn.Sequential(*layers, nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co))
        self.s = self.shortcut(ci, co, stride, blur)
        self.se = SE(co, se, A) if se else nn.Identity()
        self.act, self.sd = A(), sd


class Bottleneck(ResBlock):
    """F = 1x1 (reduce by 4) - 3x3 - 1x1 (restore): more layers for the same parameters."""
    def __init__(self, ci, co, stride, act="relu", se=0, blur=False, sd=0.0):
        super().__init__()
        A, m = ACTS[act], co // 4
        down = blur and stride > 1
        layers = [nn.Conv2d(ci, m, 1, bias=False), nn.BatchNorm2d(m), A(),
                  nn.Conv2d(m, m, 3, 1 if down else stride, 1, bias=False), nn.BatchNorm2d(m), A()]
        layers += [Blur(m)] if down else []
        self.f = nn.Sequential(*layers, nn.Conv2d(m, co, 1, bias=False), nn.BatchNorm2d(co))
        self.s = self.shortcut(ci, co, stride, blur)
        self.se = SE(co, se, A) if se else nn.Identity()
        self.act, self.sd = A(), sd


class ShakeBlock(nn.Module):
    """Shake-Shake (Gastaldi 2017): 2 residual branches. Training mixes them with random per-image weights: alpha forward, beta backward.
    Evaluation uses the plain average."""
    def __init__(self, ci, co, stride, act="relu", se=0, blur=False, sd=0.0):
        super().__init__()
        A = ACTS[act]
        branch = lambda: nn.Sequential(nn.Conv2d(ci, co, 3, stride, 1, bias=False), nn.BatchNorm2d(co), A(),
                                       nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co))
        self.b1, self.b2 = branch(), branch()
        self.s = nn.Identity() if stride == 1 and ci == co else nn.Sequential(
            nn.AvgPool2d(stride) if stride > 1 else nn.Identity(), nn.Conv2d(ci, co, 1, bias=False), nn.BatchNorm2d(co))

    def forward(self, x):
        a, b = self.b1(x), self.b2(x)
        if not self.training:
            return self.s(x) + 0.5 * (a + b)
        n = (x.shape[0], 1, 1, 1)
        alpha, beta = torch.rand(n, device=x.device).to(a.dtype), torch.rand(n, device=x.device).to(a.dtype)
        return self.s(x) + beta * a + (1 - beta) * b + ((alpha - beta) * (a - b)).detach()


class ResNet(nn.Module):
    """3x3 stem, then stages of BasicBlocks (the first block of a stage has that stage's stride), global avg pool, linear."""
    def __init__(self, widths, blocks, p_drop=0.0, n_cls=10, strides=None, bottleneck=False, stem=None, act="relu", se=0,
                 blur=False, sd=0.0, shake=False):
        super().__init__()
        strides = strides or [1] + [2] * (len(widths) - 1)
        stem = stem or widths[0]
        layers = [nn.Conv2d(3, stem, 3, 1, 1, bias=False), nn.BatchNorm2d(stem), ACTS[act]()]
        ci, B, total, k = stem, ShakeBlock if shake else Bottleneck if bottleneck else BasicBlock, sum(blocks), 0
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):                                                # stochastic depth grows linearly with depth
                layers.append(B(ci, w, strides[i] if j == 0 else 1, act=act, se=se, blur=blur, sd=sd * k / max(1, total - 1)))
                ci, k = w, k + 1
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(ci, n_cls)

    def forward(self, x):
        return self.fc(self.drop(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)))


class ShakeDropBlock(nn.Module):
    """PyramidNet block (Han 2017) with ShakeDrop (Yamada 2019). A per-image gate decides if the branch is shaken: alpha forward, beta backward.
    The shortcut has no parameters: average pool, then zero padding for the new channels."""
    def __init__(self, ci, co, stride, act, p):
        super().__init__()
        self.f = nn.Sequential(nn.BatchNorm2d(ci), nn.Conv2d(ci, co, 3, stride, 1, bias=False), nn.BatchNorm2d(co), ACTS[act](),
                               nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co))
        self.pool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.p = p

    def forward(self, x):
        y = self.f(x)
        if self.training:
            n, d = (x.shape[0], 1, 1, 1), x.device
            g = (torch.rand(n, device=d) < self.p).to(y.dtype)
            alpha, beta = (2 * torch.rand(n, device=d) - 1).to(y.dtype), torch.rand(n, device=d).to(y.dtype)
            fw, bw = g + alpha - g * alpha, g + beta - g * beta
            y = bw * y + ((fw - bw) * y).detach()
        else:
            y = self.p * y
        s = self.pool(x)
        return y + F.pad(s, (0, 0, 0, 0, 0, y.shape[1] - s.shape[1]))


class PyramidNet(nn.Module):
    """PyramidNet: the channel count grows by a fixed step in every block. Layers = 6n + 2. ShakeDrop keep rate falls linearly to p_last."""
    def __init__(self, n=18, alpha=100, p_last=0.5, n_cls=10, act="relu", p_drop=0.0, **_):
        super().__init__()
        layers, ci, total = [nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16)], 16, 3 * n
        for k in range(total):
            co = round(16 + alpha * (k + 1) / total)
            layers.append(ShakeDropBlock(ci, co, 2 if k in (n, 2 * n) else 1, act, 1 - (k + 1) / total * (1 - p_last)))
            ci = co
        self.features = nn.Sequential(*layers, nn.BatchNorm2d(ci), ACTS[act]())
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(ci, n_cls)

    def forward(self, x):
        return self.fc(self.drop(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)))


ARCHS = {
    "r18n": dict(widths=[40, 80, 160, 320], blocks=[2, 2, 2, 2]),   # ResNet-18 layout, narrow
    "s4": dict(widths=[64, 128, 256, 320], blocks=[2, 2, 2, 1]),    # wide early stages, short last stage
    "s3": dict(widths=[64, 128, 256], blocks=[3, 3, 3]),            # 3 stages, deeper, no 4x4 stage
    "s4w": dict(widths=[72, 144, 256, 336], blocks=[2, 2, 2, 1]),   # s4 widened to use the budget (4.88M)
    "r18w": dict(widths=[96, 192, 384, 768], blocks=[2, 2, 2, 2]),  # TEACHER only (25M params, no budget)
    "s4hr": dict(widths=[64, 128, 256, 320], blocks=[2, 2, 2, 1], strides=[1, 2, 1, 2]),  # s4 params, stage 3 at 16x16: ~2x compute
    "d3": dict(widths=[64, 128, 224, 288], blocks=[3, 3, 3, 1]),    # more layers: 22
    "wrn28": dict(widths=[56, 112, 224], blocks=[4, 4, 4]),         # more layers: 26, 3 stages
    "wrn40x": dict(widths=[44, 88, 176], blocks=[6, 6, 6]),         # more layers: 38, 3 stages
    "r50n": dict(widths=[112, 224, 448, 896], blocks=[3, 4, 6, 3], bottleneck=True, stem=32),  # more layers: 50, bottleneck
    "wide3": dict(widths=[96, 192, 320], blocks=[2, 2, 2]),         # more neurons per layer, fewer layers: 14
    "rebal": dict(widths=[64, 128, 256, 192], blocks=[2, 2, 3, 1]),  # budget moved from the last block to stage 3
    "shake": dict(widths=[41, 82, 164], blocks=[4, 4, 4], stem=16, shake=True),  # Shake-Shake 26 2x41d
    "pyr": dict(n=12, alpha=125, p_last=0.5),                        # PyramidNet-74 + ShakeDrop
    "h2": dict(widths=[48, 96, 192, 232], blocks=[2, 2, 2, 1]),      # 2.50M: 2 of these fit in the 5M limit
    "q4": dict(widths=[32, 64, 128, 176], blocks=[2, 2, 2, 1]),      # 1.20M: 4 of these fit in the 5M limit
}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------- train ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="s4", choices=ARCHS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--ls", type=float, default=0.1, help="label smoothing")
    ap.add_argument("--cutout", type=int, default=16)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--ta", action="store_true", help="TrivialAugment-style colour/geometry ops")
    ap.add_argument("--cutmix", type=float, default=0.0, help="probability of CutMix per batch")
    ap.add_argument("--teacher", default="", help="knowledge distillation: arch:path.pt[,arch:path.pt] (ensemble)")
    ap.add_argument("--kd_a", type=float, default=0.9, help="weight of the distillation loss")
    ap.add_argument("--kd_t", type=float, default=4.0, help="distillation temperature")
    ap.add_argument("--sam", type=float, default=0.0, help="SAM radius rho (0 = plain SGD); costs 2 passes per step")
    ap.add_argument("--tta", type=int, default=2, choices=[2, 10], help="test views: flip, or flip x 5 shifts")
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    ap.add_argument("--full", action="store_true", help="train on all 50k (no validation split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--bench", action="store_true", help="time a few steps of every arch and exit")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many training images")
    ap.add_argument("--device", default=None)
    ap.add_argument("--act", default="relu", choices=ACTS, help="activation function")
    ap.add_argument("--se", type=int, default=0, help="squeeze-and-excitation reduction ratio (0 = off)")
    ap.add_argument("--blur", action="store_true", help="anti-aliased downsampling and pooled shortcuts")
    ap.add_argument("--sd", type=float, default=0.0, help="stochastic depth: drop rate of the last block")
    ap.add_argument("--zoom", action="store_true", help="random scale augmentation")
    ap.add_argument("--augmix", action="store_true", help="AugMix-style blend instead of one TrivialAugment op")
    ap.add_argument("--lowres", type=float, default=0.0, help="probability of a random low-pass filter on a training image")
    ap.add_argument("--ema", type=float, default=0.0, help="also keep an average of the weights with this decay")
    ap.add_argument("--split", default="", help="npz with train, val_r, val_c index arrays")
    ap.add_argument("--finish_by", type=float, default=0, help="unix time: after 5 epochs, shorten the schedule so the run ends by then")
    ap.add_argument("--no_test", action="store_true", help="never load the test images (experiments and model selection)")
    a = ap.parse_args()
    global DEV
    DEV = a.device or DEV
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    amp = DEV in ("cuda", "mps")
    scaler = torch.amp.GradScaler(DEV, enabled=amp)

    if a.bench:
        for k, cfg in ARCHS.items():
            m = (PyramidNet if "alpha" in cfg else ResNet)(**cfg).to(DEV)
            opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
            x = torch.randn(a.bs, 3, 32, 32, device=DEV); y = torch.randint(0, 10, (a.bs,), device=DEV)
            for i in range(13):
                if i == 3:
                    torch.mps.synchronize() if DEV == "mps" else torch.cuda.synchronize() if DEV == "cuda" else None
                    t = time.time()
                with torch.autocast(DEV, dtype=torch.float16, enabled=amp):
                    loss = F.cross_entropy(m(augment(x)), y)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            torch.mps.synchronize() if DEV == "mps" else torch.cuda.synchronize() if DEV == "cuda" else None
            ips = 10 * a.bs / (time.time() - t)
            print(f"{k:5s} params {n_params(m):,}  {ips:,.0f} img/s  ~{50000 / ips:.0f} s/epoch")
        return

    tags = ("_ta" if a.ta else "") + (f"_cm{a.cutmix:g}" if a.cutmix else "") + ("_kd" if a.teacher else "") + \
        (f"_sam{a.sam:g}" if a.sam else "")
    name = a.name or f"{a.arch}{tags}_e{a.epochs}{'_full' if a.full else ''}_s{a.seed}"
    os.makedirs(RUNS, exist_ok=True)
    Net = PyramidNet if "alpha" in ARCHS[a.arch] else ResNet
    model = Net(**ARCHS[a.arch], p_drop=a.drop, act=a.act, se=a.se, blur=a.blur, sd=a.sd).to(DEV)
    if DEV == "cuda":
        model = model.to(memory_format=torch.channels_last)                   # faster convs on tensor cores
    P = n_params(model)
    assert P <= 5_000_000 or a.arch == "r18w", P

    teachers = []
    for spec in filter(None, a.teacher.split(",")):
        arch, path = spec.split(":", 1)
        t = ResNet(**ARCHS[arch]).to(DEV)
        t.load_state_dict(torch.load(path, map_location=DEV))
        t.eval().requires_grad_(False)
        teachers.append(t.to(memory_format=torch.channels_last) if DEV == "cuda" else t)
        print(f"teacher {arch} from {path}", flush=True)
    print(f"{name}: {P:,} trainable params on {DEV}", flush=True)

    xtr, ytr, _ = load("train")
    perm = np.random.default_rng(0).permutation(len(xtr))       # same split for every run
    tr, va = (perm, perm[:0]) if a.full else (perm[5000:], perm[:5000])
    n_r = len(va)
    if a.split:                                                 # val_r: random images. val_c: whole clusters kept out of training
        z = np.load(a.split)
        tr, va, n_r = z["train"], np.concatenate([z["val_r"], z["val_c"]]), len(z["val_r"])
    if a.limit:
        tr, va = tr[:a.limit], va[:a.limit // 4]
    to_dev = lambda arr: torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(DEV)
    Xtr, Ytr = to_dev(xtr[tr]), torch.from_numpy(ytr[tr]).to(DEV)
    Xva, Yva = to_dev(xtr[va]), torch.from_numpy(ytr[va]).to(DEV)
    if not a.no_test:
        xte, _, ids = load("test")
        Xte = to_dev(xte)
    mean = Xtr.float().mean((0, 2, 3), keepdim=True) / 255
    std = Xtr.float().std((0, 2, 3), keepdim=True) / 255
    norm = lambda xb: (xb.float() / 255 - mean) / std

    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]                 # BN and bias: no weight decay
    opt = torch.optim.SGD([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0}],
                          lr=a.lr, momentum=0.9, nesterov=True)
    steps_per_ep = len(Xtr) // a.bs
    total = a.epochs * steps_per_ep
    warm = max(1, int(a.warmup * total))
    lr_at = lambda t: a.lr * t / warm if t < warm else a.lr * 0.5 * (1 + math.cos(math.pi * (t - warm) / (total - warm)))

    @torch.no_grad()
    def predict(X, tta=2, net=None):
        """Logits averaged over test-time views: the image and its flip, optionally at 4 extra 2-pixel shifts."""
        net = net or model
        model.eval()
        shifts = [(0, 0)] if tta == 2 else [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
        out = []
        for i in range(0, len(X), 1000):
            xp = F.pad(norm(X[i:i + 1000]), (2, 2, 2, 2), mode="reflect")
            acc = 0
            for dy, dx in shifts:
                xb = xp[:, :, 2 + dy:34 + dy, 2 + dx:34 + dx]
                with torch.autocast(DEV, dtype=torch.float16, enabled=amp):
                    acc = acc + net(xb).float() + net(xb.flip(3)).float()
            out.append(acc / (2 * len(shifts)))
        model.train()
        return torch.cat(out)

    ema = copy.deepcopy(model).eval().requires_grad_(False) if a.ema else None
    hist, t0, step = [], time.time(), 0
    for ep in range(a.epochs):
        idx = torch.randperm(len(Xtr), device=DEV)
        tl, tc = 0.0, 0
        for b in range(steps_per_ep):
            j = idx[b * a.bs:(b + 1) * a.bs]
            x01 = Xtr[j].float() / 255
            if a.lowres:
                x01 = lowpass(x01, a.lowres)
            if a.zoom:
                x01 = zoom(x01)
            if a.augmix:
                x01 = augmix(x01)
            elif a.ta:
                x01 = trivial_aug(x01)
            xb, yb = augment((x01 - mean) / std, cutout=a.cutout), Ytr[j]
            lam, yb2 = 1.0, yb
            if a.cutmix and np.random.rand() < a.cutmix:                      # paste a box from another image
                lam = np.random.beta(1.0, 1.0)
                rp = torch.randperm(len(xb), device=DEV)
                h = int(32 * math.sqrt(1 - lam))
                cy, cx = np.random.randint(32), np.random.randint(32)
                y1, y2, x1, x2 = max(cy - h // 2, 0), min(cy + h // 2, 32), max(cx - h // 2, 0), min(cx + h // 2, 32)
                xb[:, :, y1:y2, x1:x2] = xb[rp, :, y1:y2, x1:x2]
                lam, yb2 = 1 - (y2 - y1) * (x2 - x1) / 1024, yb[rp]
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            if teachers:
                with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=amp):
                    t_logits = sum(t(xb).float() for t in teachers) / len(teachers)

            def loss_fn():
                with torch.autocast(DEV, dtype=torch.float16, enabled=amp):
                    logits = model(xb)
                    loss = lam * F.cross_entropy(logits, yb, label_smoothing=a.ls) + \
                        (1 - lam) * F.cross_entropy(logits, yb2, label_smoothing=a.ls)
                    if teachers:                                              # match the teachers' soft labels
                        kd = F.kl_div(F.log_softmax(logits.float() / a.kd_t, 1), F.softmax(t_logits / a.kd_t, 1),
                                      reduction="batchmean") * a.kd_t ** 2
                        loss = (1 - a.kd_a) * loss + a.kd_a * kd
                return loss

            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            scaler.scale(loss).backward()
            if a.sam:                       # SAM: move the weights rho along the gradient, take the gradient there
                params = [p for p in model.parameters() if p.grad is not None]
                inv = 1.0 / scaler.get_scale()                                # grads are still scaled
                gn = torch.norm(torch.stack([p.grad.norm() for p in params])) * inv
                if torch.isfinite(gn):                                        # on overflow the scaler skips the step
                    with torch.no_grad():
                        eps = [p.grad * (inv * a.sam / (gn + 1e-12)) for p in params]
                        torch._foreach_add_(params, eps)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss_fn()).backward()
                    with torch.no_grad():
                        torch._foreach_sub_(params, eps)
            scaler.step(opt)
            scaler.update()
            step += 1
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - min(a.ema, step / (step + 10)))
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            if b % 20 == 0:
                tl += loss.item(); tc += 1
        va_acc = (predict(Xva).argmax(1) == Yva).float().mean().item() if len(va) else float("nan")
        hist.append({"epoch": ep + 1, "train_loss": tl / tc, "val_acc": va_acc, "time_s": time.time() - t0})
        print(f"ep {ep + 1:3d}/{a.epochs}  loss {tl / tc:.3f}  val {100 * va_acc:.2f}%  {time.time() - t0:.0f}s", flush=True)
        if a.finish_by and ep == 4:                                           # fit the cosine schedule to the time that is left
            per_ep = (time.time() - t0) / 5
            a.epochs = max(6, min(a.epochs, int((a.finish_by - t0) / per_ep)))
            total = a.epochs * steps_per_ep
            warm = max(1, int(a.warmup * total))
            print(f"{per_ep:.1f} s per epoch, schedule set to {a.epochs} epochs", flush=True)
        if ep + 1 >= a.epochs:
            break

    final = {}
    if len(va):
        for k in (2, 10):
            final[f"val_acc_tta{k}"] = (predict(Xva, k).argmax(1) == Yva).float().mean().item()
        lv = predict(Xva, 2)
        hit = lv.argmax(1) == Yva
        final["val_r"], final["val_c"] = hit[:n_r].float().mean().item(), (hit[n_r:].float().mean().item() if len(va) > n_r else None)
        np.save(os.path.join(RUNS, f"{name}_val_logits.npy"), lv.cpu().numpy())
        small = F.interpolate(F.avg_pool2d(Xva.float(), 2), scale_factor=2, mode="bilinear", align_corners=False)  # no fine detail
        final["val_blur"] = (predict(small, 2).argmax(1) == Yva).float().mean().item()
        if ema is not None:
            le = predict(Xva, 2, ema)
            hit = le.argmax(1) == Yva
            final["ema_val_r"], final["ema_val_c"] = hit[:n_r].float().mean().item(), (hit[n_r:].float().mean().item() if len(va) > n_r else None)
            np.save(os.path.join(RUNS, f"{name}_ema_val_logits.npy"), le.cpu().numpy())
        print("final val:", {k: (f"{100 * v:.2f}%" if v is not None else "-") for k, v in final.items()}, flush=True)
    if a.no_test:
        torch.save(model.state_dict(), os.path.join(RUNS, f"{name}.pt"))
        json.dump({"name": name, "args": vars(a), "params": P, "device": DEV, "history": hist, "final": final},
                  open(os.path.join(RUNS, f"{name}.json"), "w"), indent=1)
        return
    other = 10 if a.tta == 2 else 2                                           # also keep the other TTA, to choose later
    te_other = predict(Xte, other).cpu()
    np.save(os.path.join(RUNS, f"{name}_test_logits_tta{other}.npy"), te_other.numpy())
    with open(os.path.join(RUNS, f"submission_{name}_tta{other}.csv"), "w") as f:
        f.write("id,label\n" + "".join(f"{i},{p}\n" for i, p in zip(ids, te_other.argmax(1).numpy())))
    te_logits = predict(Xte, a.tta).cpu()
    pred = te_logits.argmax(1).numpy()
    sub = os.path.join(RUNS, f"submission_{name}.csv")
    with open(sub, "w") as f:
        f.write("id,label\n" + "".join(f"{i},{p}\n" for i, p in zip(ids, pred)))
    torch.save(model.state_dict(), os.path.join(RUNS, f"{name}.pt"))
    np.save(os.path.join(RUNS, f"{name}_test_logits.npy"), te_logits.numpy())
    json.dump({"name": name, "args": vars(a), "params": P, "device": DEV, "history": hist, "final": final,
               "test_pred_counts": np.bincount(pred, minlength=10).tolist()},
              open(os.path.join(RUNS, f"{name}.json"), "w"), indent=1)
    print(f"wrote {sub}  test class counts {np.bincount(pred, minlength=10).tolist()}", flush=True)


if __name__ == "__main__":
    main()
