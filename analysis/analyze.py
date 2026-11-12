"""Where does the model go wrong? Capacity, depth, errors, novelty, context and frequency. TRAINING images only: no test image is loaded."""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from train import ARCHS, ResNet  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CLS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CK = {os.path.basename(p)[:-3]: p for p in glob.glob(os.path.join(ROOT, "out_*/**/*.pt"), recursive=True)}
# name -> (arch, label). All use the same 45k/5k split.
MODELS = {"s4_ta_sam0.05_e300_s0": ("s4", "SAM 0.05"), "s4_ta_e300_s0": ("s4", "TA only"), "s4_ta_sam0.1_e300_s0": ("s4", "SAM 0.1"),
          "s4_ta_kd_sam0.05_e300_s0": ("s4", "KD+SAM"), "s4_ta_kd_e300_s0": ("s4", "KD"), "s4_ta_cm0.5_e300_s0": ("s4", "CutMix"),
          "s4w_ta_e300_s0": ("s4w", "wider 4.88M"), "r18w_ta_cm0.5_e200_s0": ("r18w", "25M teacher")}
OUT = {}


def say(*a):
    print(*a, flush=True)


z = np.load(os.path.join(ROOT, "data.noindex/train.npz"))
X, Y = torch.from_numpy(z["x"]).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(z["y"]).long()
perm = np.random.default_rng(0).permutation(len(X))
tr, va = torch.from_numpy(perm[5000:]), torch.from_numpy(perm[:5000])
mean = X[tr].float().mean((0, 2, 3), keepdim=True) / 255
std = X[tr].float().std((0, 2, 3), keepdim=True) / 255
norm = lambda xb: (xb.float() / 255 - mean.to(xb.device)) / std.to(xb.device)
Yv = Y[va]


def load(name):
    m = ResNet(**ARCHS[MODELS[name][0]])
    m.load_state_dict(torch.load(CK[name], map_location="cpu", weights_only=True))
    return m.to(DEV).eval()


@torch.no_grad()
def predict(m, images, tf=None, bs=500):
    """Logits averaged over the image and its flip. tf changes the normalized batch (masks, filters)."""
    out = []
    for i in range(0, len(images), bs):
        xb = norm(images[i:i + bs].to(DEV))
        if tf is not None:
            xb = tf(xb)
        out.append(((m(xb) + m(xb.flip(3))) / 2).cpu())
    return torch.cat(out)


acc = lambda lg, y: 100 * (lg.argmax(1) == y).float().mean().item()
avail = [n for n in MODELS if n in CK]
say("device", DEV, "| models", len(avail), "| missing", [n for n in MODELS if n not in CK])

# ---------- A. Too small, or not general enough? ----------
say("\n== A. train vs held-out (clean images, flip TTA) ==")
sub = tr[:10000]
val_logits, A = {}, {}
for n in avail:
    m = load(n)
    val_logits[n] = predict(m, X[va])
    atr, ava = acc(predict(m, X[sub]), Y[sub]), acc(val_logits[n], Yv)
    nll = F.cross_entropy(val_logits[n], Yv).item()
    A[n] = dict(train=atr, val=ava, nll=nll, params=sum(p.numel() for p in m.parameters()))
    say(f"{MODELS[n][1]:14s} params {A[n]['params'] / 1e6:5.2f}M  train {atr:6.2f}  held-out {ava:6.2f}  gap {atr - ava:5.2f}  NLL {nll:.3f}")
    del m
OUT["A"] = A

# ---------- B. Which images and classes fail? Are the errors shared? ----------
base = "s4_ta_sam0.05_e300_s0"
lg = val_logits[base]
pred = lg.argmax(1)
say("\n== B. errors of the best model ==")
conf = torch.zeros(10, 10, dtype=torch.long)
for t, p in zip(Yv, pred):
    conf[t, p] += 1
per = {CLS[i]: round(100 * conf[i, i].item() / conf[i].sum().item(), 2) for i in range(10)}
say("per class:", per)
pairs = sorted(((conf[i, j].item() + conf[j, i].item(), CLS[i], CLS[j]) for i in range(10) for j in range(i + 1, 10)), reverse=True)
errs = (pred != Yv).sum().item()
say("errors", errs, "| top pairs:", [(c, a, b) for c, a, b in pairs[:6]], "| share of cat-dog", round(100 * pairs[0][0] / errs, 1), "%")
s4names = [n for n in avail if MODELS[n][0] == "s4"]
wrong = torch.stack([val_logits[n].argmax(1) != Yv for n in s4names]).sum(0)
say(f"over {len(s4names)} s4 models: images wrong in ANY {(wrong > 0).sum().item()}, wrong in ALL {(wrong == len(s4names)).sum().item()}, wrong in exactly 1 {(wrong == 1).sum().item()}")
ens = sum(F.softmax(val_logits[n], 1) for n in s4names)
ens2 = F.softmax(val_logits[base], 1) + F.softmax(val_logits["s4_ta_sam0.1_e300_s0"], 1)
say(f"ensemble (not submittable, it only measures variance): all s4 {acc(ens, Yv):.2f}, best 2 {acc(ens2, Yv):.2f}, single best {acc(lg, Yv):.2f}")
OUT["B"] = dict(per_class=per, pairs=pairs[:6], errors=errs, wrong_any=(wrong > 0).sum().item(), wrong_all=(wrong == len(s4names)).sum().item(),
                ens_all=acc(ens, Yv), ens2=acc(ens2, Yv))


# ---------- C. Does the model fail on images unlike the training set? ----------
@torch.no_grad()
def feats(m, images, bs=500):
    out = []
    for i in range(0, len(images), bs):
        out.append(F.adaptive_avg_pool2d(m.features(norm(images[i:i + bs].to(DEV))), 1).flatten(1).cpu())
    return torch.cat(out)


say("\n== C. novelty: accuracy by distance to the nearest training image ==")
m = load(base)
t0 = time.time()
Ftr, Fva = F.normalize(feats(m, X[tr]), dim=1), F.normalize(feats(m, X[va]), dim=1)
sim = torch.cat([(Fva[i:i + 500].to(DEV) @ Ftr.to(DEV).T).amax(1).cpu() for i in range(0, 5000, 500)])
Ptr = X[tr].float().flatten(1) / 255
Pva = X[va].float().flatten(1) / 255
dpx = torch.cat([torch.cdist(Pva[i:i + 200], Ptr).amin(1) for i in range(0, 5000, 200)])
say(f"nearest neighbours in {time.time() - t0:.0f}s")
ok = (pred == Yv).float()
C = {}
for key, score in (("pixel distance", dpx), ("feature distance", 1 - sim)):
    q = torch.quantile(score, torch.tensor([.2, .4, .6, .8]))
    b = torch.bucketize(score, q)
    row = [round(100 * ok[b == k].mean().item(), 2) for k in range(5)]
    say(f"{key:18s} quintile near->far: {row}")
    C[key] = row
for t in (2.0, 3.0, 4.0):
    k = dpx < t
    say(f"pixel L2 < {t}: {k.sum().item():4d} images, acc {100 * ok[k].mean().item() if k.any() else 0:.2f} | others acc {100 * ok[~k].mean().item():.2f}")
OUT["C"] = C

# ---------- D. Object or background? Which frequencies? ----------
say("\n== D. what the models use: acc on masked and filtered held-out images ==")
yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
center = ((yy >= 8) & (yy < 24) & (xx >= 8) & (xx < 24)).float()[None, None].to(DEV)
r = torch.fft.fftshift(torch.sqrt((yy - 16.0) ** 2 + (xx - 16.0) ** 2)).to("cpu")


def band(x, lo, hi):
    """Keep Fourier radii in [lo, hi). FFT runs on the CPU because MPS FFT support varies."""
    f = torch.fft.fft2(x.cpu())
    keep = ((r >= lo) & (r < hi)).float()
    return torch.fft.ifft2(f * keep).real.to(DEV)


TESTS = {"clean": None, "background only (center 16x16 hidden)": lambda x: x * (1 - center), "center only (border hidden)": lambda x: x * center,
         "low-pass r<8": lambda x: band(x, 0, 8), "low-pass r<4": lambda x: band(x, 0, 4), "high-pass r>=4": lambda x: band(x, 4, 99),
         "gray": lambda x: x.mean(1, keepdim=True).expand_as(x)}
D = {}
for n in [k for k in (base, "s4_ta_e300_s0", "r18w_ta_cm0.5_e200_s0") if k in avail]:
    mm = load(n)
    D[n] = {k: round(acc(predict(mm, X[va], tf), Yv), 2) for k, tf in TESTS.items()}
    say(MODELS[n][1], D[n])
    del mm
OUT["D"] = D

# ---------- E. Is capacity used? Which layer adds what? ----------
say("\n== E. layer by layer, best model ==")
blocks = {3: "stage1.1", 4: "stage1.2", 5: "stage2.1", 6: "stage2.2", 7: "stage3.1", 8: "stage3.2", 9: "stage4.1"}
store = {}
hooks = [m.features[i].register_forward_hook(lambda mod, inp, o, i=i: store.__setitem__(i, F.adaptive_avg_pool2d(o, 1).flatten(1).detach().cpu())) for i in blocks]


@torch.no_grad()
def layer_feats(images, bs=500):
    res = {i: [] for i in blocks}
    for k in range(0, len(images), bs):
        m(norm(images[k:k + bs].to(DEV)))
        for i in blocks:
            res[i].append(store[i])
    return {i: torch.cat(v) for i, v in res.items()}


Ltr, Lva = layer_feats(X[tr[:20000]]), layer_feats(X[va])
for h in hooks:
    h.remove()
ytr20 = Y[tr[:20000]]


def probe(a, b, steps=300):
    """Linear classifier on one layer's pooled features: how much of the task is solved at this depth?"""
    mu, sd = a.mean(0), a.std(0) + 1e-6
    a, b = ((a - mu) / sd).to(DEV), ((b - mu) / sd).to(DEV)
    w = nn.Linear(a.shape[1], 10).to(DEV)
    opt = torch.optim.Adam(w.parameters(), lr=0.02, weight_decay=1e-4)
    y = ytr20.to(DEV)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(w(a), y).backward()
        opt.step()
    return 100 * (w(b).argmax(1).cpu() == Yv).float().mean().item()


E = {}
for i, nm in blocks.items():
    f = Lva[i]
    mu = f.mean(0)
    cov = torch.cov((f - mu).T.cpu())
    ev = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)
    cum = ev.cumsum(0) / ev.sum()
    r95, r99 = (cum < .95).sum().item() + 1, (cum < .99).sum().item() + 1
    alive = (mu > 0.01 * mu.max()).sum().item()
    p = probe(Ltr[i], f)
    E[nm] = dict(width=f.shape[1], alive=alive, rank95=r95, rank99=r99, probe=round(p, 2))
    say(f"{nm}: width {f.shape[1]:3d}  active channels {alive:3d}  dims for 95%/99% of variance {r95:3d}/{r99:3d}  linear probe {p:6.2f}")


class Skip(nn.Module):
    """Removes a block's residual branch: the block then only passes its input through ReLU."""
    def forward(self, x):
        return F.relu(x)


F_full = acc(lg, Yv)
for i, nm in ((3, "stage1.1"), (4, "stage1.2"), (6, "stage2.2"), (8, "stage3.2")):
    keep = m.features[i]
    m.features[i] = Skip()
    a = acc(predict(m, X[va]), Yv)
    m.features[i] = keep
    E[nm]["knockout"] = round(a, 2)
    say(f"remove {nm}: held-out {a:.2f} (full {F_full:.2f})")
OUT["E"] = E

# BN scale of channels and params per stage
stage_params = {}
for nm, mod in m.named_children():
    pass
for i, nm in blocks.items():
    stage_params[nm] = sum(p.numel() for p in m.features[i].parameters())
say("params per block:", {k: f"{v / 1e6:.2f}M" for k, v in stage_params.items()})
OUT["E_params"] = stage_params

# ---------- F. Grad-CAM on the most confident errors ----------
say("\n== F. Grad-CAM figure ==")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

p = F.softmax(lg, 1)
top = p.amax(1)
wrong_idx = torch.nonzero(pred != Yv).flatten()
pick = wrong_idx[top[wrong_idx].argsort(descending=True)[:40]]
cap = {}
h1 = m.features[8].register_forward_hook(lambda mod, i, o: (cap.__setitem__("a", o), o.retain_grad()))
fig, ax = plt.subplots(8, 10, figsize=(15, 13))
for k, idx in enumerate(pick.tolist()):
    img = X[va[idx]][None]
    xb = norm(img.to(DEV)).requires_grad_(True)
    out = m(xb)
    m.zero_grad()
    out[0, pred[idx]].backward()
    a = cap["a"]
    w = a.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, (32, 32), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
    cam = cam / (cam.max() + 1e-6)
    row, col = k // 5, (k % 5) * 2
    ax[row, col].imshow(img[0].permute(1, 2, 0).numpy(), interpolation="nearest")
    ax[row, col].set_title(f"true {CLS[Yv[idx]]}", fontsize=7)
    ax[row, col + 1].imshow(img[0].permute(1, 2, 0).numpy(), interpolation="nearest")
    ax[row, col + 1].imshow(cam, cmap="jet", alpha=0.5, interpolation="bilinear")
    ax[row, col + 1].set_title(f"said {CLS[pred[idx]]} {100 * top[idx].item():.0f}%", fontsize=7)
for a_ in ax.flat:
    a_.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "gradcam_errors.png"), dpi=110)
h1.remove()
say("saved gradcam_errors.png")
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
say("DONE")
