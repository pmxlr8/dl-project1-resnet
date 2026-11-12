# A 4.5M-parameter ResNet for CIFAR-10

NYU CS-GY 6953 Deep Learning, Fall 2026, Project 1. Kaggle team **dodo**: Pranjal Mishra (pm4084) and Devansh Vikram (dv2482).

Competition: [cifar-10-res-net-architecture-challenge-dl](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl). The task is to train a residual network with at most 5 million trainable parameters from scratch on CIFAR-10.

## Final result

| | |
|---|---|
| Model | `s4` ResNet: widths 64-128-256-320, blocks 2-2-2-1, basic blocks, 3x3 stem, global average pooling, linear head |
| Trainable parameters | **4,521,034** |
| Recipe | SGD (Nesterov, momentum 0.9), lr 0.2, batch 256, wd 5e-4, 5% warmup + cosine, label smoothing 0.1, crop + flip + Cutout (`--cutout 16`, a 15x15 hole), TrivialAugment, SAM rho 0.05, 300 epochs, all 50,000 training images, seed 2 |
| Test-time | average of 10 views (image + flip, at the original position and 4 shifts of 2 px) |
| Held-out accuracy (same recipe, seed 0, 45,000 / 5,000 split) | 97.66% (flip views) |
| **Kaggle test accuracy** | **public 89.3%, private 85.6%** (1,000 images each; 87.45% over all 2,000) |

The report explains the 8 to 12 point gap between held-out and test accuracy, and lists all experiments.

## Repository layout

| Path | Content |
|---|---|
| `train.py` | Trains every model in the report tables. Model definitions (`ARCHS`), GPU augmentation, SAM, distillation, test-time views, submission writer. |
| `notebooks/results.ipynb` | Executed notebook: prints the parameter count, recomputes the held-out accuracy and the final test predictions from the checkpoints, prints the Kaggle test accuracy, plots training curves and all scores. |
| `checkpoints/` | `s4_final_sam_seed2_full50k.pt` (the final submitted model) and `s4_sam_seed0_train45k.pt` (same recipe, trained on 45,000 images, for the held-out check). |
| `results/submission_final.csv` | The file that Kaggle scored for the final model. |
| `results/kaggle_submissions.csv` | All our submissions with public and private scores. |
| `logs/runs/*.json` | Arguments, parameter count and per-epoch history of every run in the report. |
| `logs/*.log` | Full training logs of the final run and of the held-out run. |
| `kaggle_notebooks/` | The Kaggle notebooks that trained the final model (`final_run_v5.ipynb`) and the held-out model (`heldout_run_v4.ipynb`). |
| `analysis/` | `make_split.py` (grouped held-out split), `split_x.npz` (its indices), `analyze.py` (error analysis) and its output, `specialist.py` and `specialist_eval.py` (cat-vs-dog specialist). These scripts need the checkpoints of the compared runs, which are not all in this repository. |

## Reproduce

```bash
pip install -r requirements.txt

# 1. Data (needs a Kaggle account that joined the competition)
kaggle competitions download -c cifar-10-res-net-architecture-challenge-dl -p data.noindex
unzip -q data.noindex/cifar-10-res-net-architecture-challenge-dl.zip -d data.noindex
# train.py decodes the PNGs once and caches them as data.noindex/train.npz and test.npz

# 2. Final model: about 2 h on one NVIDIA T4. Writes runs/submission_<name>.csv and runs/<name>.pt
python train.py --arch s4 --epochs 300 --ta --tta 10 --sam 0.05 --full --seed 2

# 3. Held-out check of the same recipe: trains on 45,000 images and reports accuracy on the other 5,000
python train.py --arch s4 --epochs 300 --ta --tta 10 --sam 0.05 --seed 0

# 4. Print the parameter count only
python -c "from train import ResNet, ARCHS; m = ResNet(**ARCHS['s4']); print(sum(p.numel() for p in m.parameters() if p.requires_grad))"
```

Other runs in the report use the same script with other flags. The exact arguments of every run are in `logs/runs/<run>.json` under `args`. Examples:

| Run | Flags |
|---|---|
| Baseline (no TrivialAugment) | `--arch s4 --epochs 200 --full` |
| Distillation | `--arch s4 --epochs 300 --ta --teacher r18w:<path>,s4:<path>,s4:<path> --full` |
| 600 epochs | `--arch s4 --epochs 600 --ta --tta 10 --sam 0.05 --full --seed 4` |
| Grouped-split screen | `--arch s4 --epochs 100 --ta --split analysis/split_x.npz --act silu` |
| Shake-Shake / PyramidNet | `--arch shake` / `--arch pyr` with `--ta --wd 1e-4` |

GPU runs are not bit-for-bit deterministic, so a new run can differ from ours by a few tenths of a point.

## Rules we followed

We used only the 50,000 provided training images. No pretrained weights. The hidden test images were never used for training, validation or model selection; the only feedback from them was the official Kaggle score. Every submitted model is a single network under 5M parameters.

## References

The methods we used are cited in the report: ResNet (He et al. 2016), TrivialAugment (Muller and Hutter 2021), Cutout (DeVries and Taylor 2017), label smoothing (Szegedy et al. 2016), SAM (Foret et al. 2021), cosine schedule (Loshchilov and Hutter 2017), distillation (Hinton et al. 2015), Shake-Shake (Gastaldi 2017), PyramidNet (Han et al. 2017), ShakeDrop (Yamada et al. 2019).
