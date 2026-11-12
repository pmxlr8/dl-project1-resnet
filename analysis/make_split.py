"""Build a validation split that imitates new photos. For each class, group the images by the features of a mid-level layer (stage 3), which keep visual variety, then keep
whole groups out of training. val_c = those groups, val_r = random images, train = the rest. TRAINING images only."""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from train import ARCHS, ResNet  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
z = np.load(os.path.join(ROOT, "data.noindex/train.npz"))
x, y = z["x"], z["y"]
X = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
mean, std = X.float().mean((0, 2, 3), keepdim=True) / 255, X.float().std((0, 2, 3), keepdim=True) / 255
ck = glob.glob(os.path.join(ROOT, "out_*/**/s4_ta_sam0.05_e300_full_s0.pt"), recursive=True)[0]
m = ResNet(**ARCHS["s4"])
m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
m = m.to(DEV).eval()
with torch.no_grad():
    f = torch.cat([F.adaptive_avg_pool2d(m.features[:9](((X[i:i + 500].to(DEV).float() / 255) - mean.to(DEV)) / std.to(DEV)), 1).flatten(1).cpu()
                   for i in range(0, len(X), 500)])
f = F.normalize(f, dim=1).numpy()
rng = np.random.default_rng(0)
K, TARGET = 40, 250
val_c, groups = [], []
for c in range(10):
    idx = np.nonzero(y == c)[0]
    lab = KMeans(K, n_init=3, random_state=0).fit_predict(f[idx])
    n = 0
    for g in rng.permutation(K):
        if n >= TARGET:
            break
        mem = idx[lab == g]
        val_c.append(mem)
        groups.append((c, mem))
        n += len(mem)
    print("class", c, "held-out images", n, "from", sum(1 for g in groups if g[0] == c), "groups", flush=True)
val_c = np.concatenate(val_c)
rest = np.setdiff1d(np.arange(len(x)), val_c)
rest = rng.permutation(rest)
val_r, train = rest[:2500], rest[2500:]
assert not (set(val_c) & set(val_r)) and not (set(val_c) & set(train)) and not (set(val_r) & set(train))
np.savez_compressed(os.path.join(HERE, "split_x.npz"), train=train.astype(np.int32), val_r=val_r.astype(np.int32), val_c=val_c.astype(np.int32))
print("train", len(train), "val_r", len(val_r), "val_c", len(val_c), "file KB", os.path.getsize(os.path.join(HERE, "split_x.npz")) // 1024)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

show = [g for c in range(10) for g in [gg for gg in groups if gg[0] == c][:1]]
fig, ax = plt.subplots(10, 14, figsize=(12, 9))
for r, (c, mem) in enumerate(show):
    for k in range(14):
        ax[r, k].imshow(x[mem[k]], interpolation="nearest")
        ax[r, k].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "clusters.png"), dpi=90)
print("DONE")
