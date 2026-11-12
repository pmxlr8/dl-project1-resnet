"""Cat-vs-dog specialist: a small ResNet trained only on cat and dog training images (new-photo split). It must fit in the 478,966 parameters left beside s4.
Training images only. No test image is loaded. Usage: specialist.py <seed> <epochs>"""
import math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import ResNet, augment, trivial_aug
seed, epochs = int(sys.argv[1]), int(sys.argv[2])
torch.manual_seed(seed); np.random.seed(seed)
DEV, BS, CAT, DOG = "mps", 128, 3, 5
z = np.load("interp/split_x.npz"); d = np.load("data.noindex/train.npz"); x, y = d["x"], d["y"]
tr = z["train"]; va = np.concatenate([z["val_r"], z["val_c"]])
X = torch.from_numpy(x).permute(0, 3, 1, 2)
mean = (X[tr].float().mean((0, 2, 3), keepdim=True) / 255).to(DEV); std = (X[tr].float().std((0, 2, 3), keepdim=True) / 255).to(DEV)
keep = tr[np.isin(y[tr], [CAT, DOG])]
Xt = X[keep].to(DEV); Yt = torch.from_numpy((y[keep] == DOG).astype(np.int64)).to(DEV)
m = ResNet([20, 40, 80, 112], [2, 2, 2, 1], n_cls=2, act="silu").to(DEV)
P = sum(p.numel() for p in m.parameters()); assert P <= 478_966; print("params", P, "train images", len(keep), flush=True)
decay = [p for p in m.parameters() if p.ndim > 1]; rest = [p for p in m.parameters() if p.ndim <= 1]
opt = torch.optim.SGD([{"params": decay, "weight_decay": 5e-4}, {"params": rest, "weight_decay": 0}], lr=0.1, momentum=0.9, nesterov=True)
n = len(Xt) // BS; total = epochs * n; warm = int(0.05 * total); t0 = time.time(); step = 0
for ep in range(epochs):
    idx = torch.randperm(len(Xt), device=DEV); m.train()
    for b in range(n):
        j = idx[b * BS:(b + 1) * BS]
        xb = augment((trivial_aug(Xt[j].float() / 255) - mean) / std, cutout=8)
        lr = 0.1 * step / max(1, warm) if step < warm else 0.05 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(set_to_none=True); F.cross_entropy(m(xb), Yt[j], label_smoothing=0.1).backward(); opt.step(); step += 1
    if ep % 10 == 9 or ep == 0: print(f"ep {ep + 1}/{epochs} {time.time() - t0:.0f}s", flush=True)
m.eval(); out = []
Xv = (X[va].float() / 255).to(DEV)
with torch.no_grad():
    for i in range(0, len(Xv), 500):
        xb = (Xv[i:i + 500] - mean) / std
        out.append((m(xb) + m(xb.flip(3))).cpu() / 2)
lg = torch.cat(out).numpy()
np.save(f"interp/spec_s{seed}_margin.npy", lg[:, 0] - lg[:, 1])                 # above 0 means cat
torch.save(m.state_dict(), f"interp/spec_s{seed}.pt"); print("DONE", time.time() - t0, flush=True)
