# Introduction to Transformers — AstroAI Workshop 2026

A hands-on tutorial on **transformers applied to astronomical data**, given at the 2026
[AstroAI](https://astroai.cfa.harvard.edu/) workshop at the Center for Astrophysics |
Harvard & Smithsonian.

We build two encoder-only transformers **from scratch** and train them on real ZTF data
queried through the [ALeRCE](https://alerce.science/) broker. The same `TransformerBlock`
powers both — only the tokenizer at the front changes.

| Model | Input | Task |
|-------|-------|------|
| **Sequence Transformer** | Light curves `(T=60, 2 bands)` | Classify SNIa / SNIbc / SNII / SLSN |
| **Vision Transformer (ViT)** | Image stamps `(3, 63, 63)` | Classify AGN / SN / VS |

## Contents

- [`transformers_tutorial.ipynb`](transformers_tutorial.ipynb) — the tutorial notebook.
  - **Part 1 — Dataset Creation:** queries ALeRCE for image stamps and light curves and
    builds the datasets. *Pre-run before the workshop; you do not need to run it* — the
    pre-built data ships in `dataset/`.
  - **Part 2 — Transformers:** builds attention, the transformer block, and the two models,
    then trains, evaluates (confusion matrices), and visualizes attention.
- `utils/` — supporting modules: light-curve preprocessing (`lc_preprocessing.py`),
  data loaders (`data.py`), and training/evaluation helpers (`training.py`).
- `dataset/` — pre-built `stamps.npz` and `lightcurves.npz` plus example figures.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+ recommended
pip install -r requirements.txt
jupyter notebook transformers_tutorial.ipynb
```

The models are deliberately small (d_model=64, 4 heads, 2 layers) and train on CPU in a few
minutes; a GPU is used automatically if available.

## Data & methods

- **Source:** ZTF alerts served by the ALeRCE broker. Class labels come from ALeRCE's stamp
  classifier ([Carrasco-Davis et al. 2021](https://arxiv.org/abs/2008.03309)) and light-curve
  classifier (Sánchez-Sáez et al. 2021).
- **Light curves** are irregularly sampled and interpolated onto a regular 3-day grid from
  −30 to +150 days, with a binary mask flagging real vs. filled steps.
- These are small, teaching-scale datasets (~286 stamps, ~310 light curves). The goal is to
  demonstrate the **method**, not a state-of-the-art benchmark.
