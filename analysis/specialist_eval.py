"""Does the specialist fix cat-dog errors? Compare main model alone vs main + lam * specialist margin, on own held-out images."""
import glob, sys, numpy as np
z = np.load("interp/split_x.npz"); y = np.load("data.noindex/train.npz")["y"]
vr, vc = z["val_r"], z["val_c"]; Y = np.concatenate([y[vr], y[vc]]); nr = len(vr); gr = np.arange(len(Y)) >= nr
CAT, DOG = 3, 5; cd = np.isin(Y, [CAT, DOG])
Ms = {f: np.load(f) for f in sorted(glob.glob("interp/spec_s*_margin.npy"))}
Ls = {k: np.load(p) for k, p in dict(silu="out_s1/runs/silu_val_logits.npy", silu_s1="out_s3/runs/silu_s1_val_logits.npy", base="out_s1/runs/base_val_logits.npy", base_s1="out_s1/runs/base_s1_val_logits.npy").items()}
for f, M in Ms.items():
    print(f, "specialist alone on true cat/dog: random %.2f groups %.2f" % (100 * ((M < 0) == (Y == DOG))[cd & ~gr].mean(), 100 * ((M < 0) == (Y == DOG))[cd & gr].mean()))
    for k, L in Ls.items():
        mm = L[:, CAT] - L[:, DOG]; print("  ", k, "main alone on true cat/dog, cat vs dog only: random %.2f groups %.2f" % (100 * ((mm < 0) == (Y == DOG))[cd & ~gr].mean(), 100 * ((mm < 0) == (Y == DOG))[cd & gr].mean()))
        r = L.argmax(1) == Y
        for lam in (0.1, 0.25, 0.5, 1.0, 2.0):
            N = L.copy(); N[:, CAT] += lam * M / 2; N[:, DOG] -= lam * M / 2; a = N.argmax(1) == Y
            fx, br = int((a & ~r).sum()), int((r & ~a).sum()); print(f"     lam {lam:4.2f} random {100 * a[~gr].mean():6.2f} groups {100 * a[gr].mean():6.2f} all {100 * a.mean():6.2f} | alone {100 * r.mean():6.2f} +{fx}/-{br} z{(fx - br) / max(1, fx + br) ** .5:+.1f}")
