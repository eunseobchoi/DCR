# DCR: Disagreement-Complement Refinement

DCR refines the averaged output of three frozen polyp-segmentation models (ResNet34-U-Net, DeepLabV3-ResNet50, SegFormer-B0).
In every 32×32 tile, the centred base predictions `D = [p1 − p̄, p2 − p̄, p3 − p̄]` span a subspace of rank at most 2.
A 6-channel U-Net (RGB + three probability maps) predicts two correction fields `a` and `b`.
`a` is projected onto the disagreement subspace and `b` onto its orthogonal complement, and each part is supervised by the matching component of the ground-truth residual `e = y − p̄`.

```
q = clip(p̄ + β [ P a + (I − P) b ]),    P v = Q (Qᵀ v)
```

`Q` is an orthonormal basis from a per-tile SVD of `D` on valid (non-padding) pixels. The dense `n × n` projector is never formed.

This is research code and is not intended for clinical use. The repository contains the implementation only. Trained weights and experimental results are not included.

## Installation

Python ≥ 3.10 and PyTorch ≥ 2.0.

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
pytest -q                                              # unit and integration tests (CPU, synthetic data)
dcr geometry check                                     # projection identities for rank 0/1/2 and padded tiles
dcr train refiner --method M07 --synthetic --steps 20 --out runs/demo-M07
dcr runs generate --out runs/matrix.jsonl              # full experiment matrix
dcr runs validate --input runs/matrix.jsonl --expect-total 90
```

Without installing, prefix commands with `PYTHONPATH=src python -m dcr.cli`.

## Repository layout

| Path | Contents |
|---|---|
| `src/dcr/geometry.py` | tiling, per-tile SVD basis, projection, fixed random subspaces (M06), error-capture diagnostics |
| `src/dcr/losses.py` | BCE + soft Dice, component losses normalised by `r` and `n − r`, keep loss, residual MSE |
| `src/dcr/methods.py` | the seven refinement methods behind one interface |
| `src/dcr/models/` | refiner U-Net (zero-initialised heads) and the three base models |
| `src/dcr/data/` | aspect-preserving letterbox, manifest checks, centre splits, out-of-fold contract, datasets |
| `src/dcr/evaluation/` | Dice, common-error recovery and harm, negative-frame false positives, boundary F1, paired cluster bootstrap |
| `src/dcr/registry.py` | run matrix and the one-shot external-evaluation lock |
| `src/dcr/train.py` | refiner training loop (β = 1, resumable checkpoints) |
| `configs/` | protocol settings; `null` values are fixed on the development split |

## Methods

| ID | Method | Correction |
|---|---|---|
| M01 | pixel-wise weighted fusion | `Σ wᵢ pᵢ` with softmax weights |
| M02 | disagreement subspace only | `P a` |
| M03 | complement only | `(I − P) b` |
| M04 | generic residual | `h` |
| M05 | unrestricted two-head (same network as M07) | `a + b`, residual MSE |
| M06 | same-rank random subspace | `P_rand a + (I − P_rand) b` |
| M07 | DCR | `P a + (I − P) b`, component supervision |

All methods start from `q = p̄` because the output heads are zero-initialised.

## Evaluation protocol

- PolypGen centres C1–C3 are used for training. The refiner is trained on out-of-fold base predictions: images from each centre are predicted by base models that never saw that centre.
- C4 is the development centre. It is split by cluster into `dev_select` (choosing β, threshold and loss weights) and `dev_mid` (a single check of the frozen setting).
- C5 and C6 are the external test centres. `dcr evaluate lock` freezes the model, β, threshold and post-processing, and `dcr evaluate external` then permits exactly one evaluation.
- Primary comparison: M07 against M05, centre-equal positive-image Dice. Scores are averaged over 3 seeds per image and then over each patient, sequence or near-duplicate cluster. The 95% CI comes from a paired cluster bootstrap within each centre (2,000 resamples).
- Secondary metrics: common-miss and common-false-alarm recovery, harm to correct baseline pixels, false-positive rate on negative frames, and boundary F1 at a 2-pixel tolerance.

## Data

Datasets are not redistributed. Obtain them from their providers under their licences:

- PolypGen: Ali et al., *A multi-centre polyp detection and segmentation dataset for generalisability assessment*, Scientific Data 10, 75 (2023). https://doi.org/10.1038/s41597-023-01981-y
- Kvasir-SEG: https://datasets.simula.no/kvasir-seg/
- CVC-ClinicDB: https://polyp.grand-challenge.org/CVCClinicDB/

Base-model predictions are read from `{cache_dir}/{sample_id}.npz`. Each file holds `probs` (float32 `[3, 512, 512]`, in the order unet_r34, deeplab_r50, segformer_b0) and `source_train_sites`, which is checked against the image's centre when the file is loaded.

## License

MIT. See `LICENSE`.
