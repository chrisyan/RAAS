# README

## Project Overview

RAAS is an anomaly segmentation research codebase built on top of Mask2Former and detectron2. It implements **Maskomaly** — three model variants that detect out-of-distribution objects in driving scenes using a Swin-L backbone trained on Cityscapes.

## Environment

```bash
conda activate raas   # Python 3.8, PyTorch 1.9.0, CUDA 11.1
```

Do **not** `pip install detectron2` separately. detectron2 must be built from source at `raas/detectron2/`:

```bash
cd /path/to/raas/detectron2 && pip install -e .
```

OpenAI CLIP is required for `maskomaly_id` and `maskomaly_ood`:
```bash
pip install git+https://github.com/openai/CLIP.git
# If wrong clip package is installed: pip uninstall clip -y first
```

## Running Evaluation and Inference

All scripts must be run from `Maskomaly/scripts/` with `conda activate raas`.

```bash
# Evaluation (inference + metrics in one pass)
python run_eval.py --model maskomaly     --dataset fs_laf
python run_eval.py --model maskomaly_id  --dataset smiyc_anomaly --debug
python run_eval.py --model maskomaly_ood --dataset smiyc_anomaly --strategy per_image

# Inference on an image folder (no ground truth)
python infer.py --model maskomaly --input /path/to/images --output /path/to/results
python infer.py --model maskomaly_id --input /path/to/images --output /path/to/results --debug
```

Override dataset paths at runtime to avoid editing source files:
```bash
python run_eval.py --model maskomaly --dataset fs_laf \
    --input /path/to/LostAndFound \
    --output /path/to/results
```

## Hardcoded Paths (Must Update Per Machine)

These constants appear near the top of `scripts/run_eval.py` and `scripts/infer.py`:

| Constant | Default |
|---|---|
| `_DETECTRON2_DIR` | `/home/zhiranworkstation/raas/detectron2` |
| `DEFAULT_CONFIG` | `.../Mask2Former/configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml` |
| `DEFAULT_WEIGHTS` | `.../Maskomaly/maskomaly/ckpt/model_final_17c1ee.pkl` |
| `DEFAULT_OUTPUT_BASE` | `/media/zhiranworkstation/Expansion/1_gaiax/IV2026_anomaly_segmentation` |
| `DATASET_CONFIGS` | Per-dataset root paths |

Also `sys.path.append('/home/zhiranworkstation/raas/Mask2Former')` is hardcoded inside each model file (`model_ori.py`, `model_id.py`, `model_ood.py`) — update these too.

## Architecture

### Directory Layout
```
raas/
├── detectron2/              ← detectron2 source, built with pip install -e .
├── Mask2Former/             ← Mask2Former source
└── Maskomaly/
    ├── maskomaly/
    │   ├── model_ori.py     ← maskomaly (original, hardcoded mask indices)
    │   ├── model_id.py      ← maskomaly_id (road polygon + CLIP, ID prompts)
    │   ├── model_ood.py     ← maskomaly_ood (road polygon + CLIP, ID + OOD prompts)
    │   └── datasets.py      ← Dataset classes for all 5 benchmarks
    ├── detectron2_replacements/   ← patched DefaultPredictor
    ├── mask2former_replacements/  ← patched MaskFormer model
    └── scripts/
        ├── run_eval.py      ← end-to-end inference + evaluation
        ├── infer.py         ← inference only (no ground truth)
        └── eval.py          ← metric functions (AUROC, AUPR, AP)
```

### Key Architectural Invariant: sys.path Ordering

`detectron2_replacements/` **must** appear before `raas/detectron2/` in `sys.path`. The scripts insert it first automatically. This replaces `detectron2.engine.defaults.DefaultPredictor` with a patched version that returns **3 values** instead of 1:

```python
segmentation, mask_cls_result, mask_pred_result = self.model(image)
# segmentation:      dict with "sem_seg" logits [C, H, W]
# mask_cls_result:   [N_queries, N_classes+1] — raw logits, needs softmax
# mask_pred_result:  [N_queries, H, W] — raw logits, needs sigmoid
```

If you see `ValueError: not enough values to unpack (expected 3, got 1)`, the upstream detectron2 DefaultPredictor is being picked up instead of the patch.

### Model Logic

All three models share the same `BaseSegmentationModel.get_probs_and_seg()` → softmax + sigmoid → numpy pipeline.

**`maskomaly` (model_ori.py):** Combines two signals: (1) high-entropy rejection — any query whose top class confidence > 0.7 suppresses its mask region; (2) anomaly promotion — queries at fixed indices [49, 31, 83, 32] contribute positively. Final score = `0.6 * rejection_mask + 0.4 * promotion_mask`. Note: `model_ori.py` contains leftover debug writes (`soft_mask.png`, `soft_mask2.png`, `all_soft_masks/` directory) that run unconditionally.

**`maskomaly_id` (model_id.py):** After computing the base soft mask, applies road-aware CLIP filtering: extracts road polygon from query mask #20, finds unmasked regions inside the polygon, crops each connected component, and classifies it with CLIP against 19 Cityscapes ID prompts. Components with ID confidence > 0.85 are suppressed (score → 0.05); others are marked anomalous (score → 1.0). CLIP model is loaded fresh per image call — there is no caching.

**`maskomaly_ood` (model_ood.py):** Same pipeline as `maskomaly_id` but the CLIP text prompts include additional OOD phrases ("something unusual in a driving scene", etc.). Decision logic uses combined ID+OOD probability instead of ID-only.

### Dataset Classes (`maskomaly/datasets.py`)

| Class | Dataset key | Image dir | Label dir |
|---|---|---|---|
| `FishyScapesLaF` | `fs_laf` | `original/` | `labels_masks/` |
| `FishyScapesStatic` | `fs_static` | `images/` | `labels_masks/` |
| `SMIYCANO` | `smiyc_anomaly` | `images_val/` (filtered to `validation*`) | `labels_masks/` |
| `SMIYCOBS` | `smiyc_obstacle` | `images_val/` (filtered to `validation*`) | `labels_masks/` |
| `RoadAnomaly` | `roadanomaly` | `original/` | `labels/` |

All datasets return `(image, anomaly_gt, ignore, filename)`. Anomaly ground truth uses `label == 1`; ignore/void uses `label == 255` (dataset-specific).

### Metrics (`scripts/eval.py`)

`get_scores(ground_truths, anomaly_probs, ignores, mode)` returns `(AP, AUROC, FPR@95, AUPR)`. `mode="accumulate"` (default) flattens all images together before scoring; `mode="image"` averages per-image scores.

## Acknowledgements

We thank the authors of the following codebases, which this repository builds upon:

- [Maskomaly](https://github.com/jan-ackermann/Maskomaly) — anomaly segmentation with Mask2Former
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) — universal image segmentation
- [detectron2](https://github.com/facebookresearch/detectron2) — object detection and segmentation framework

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ackermann2023maskomaly,
  title={xxx},
  author={xx},
  booktitle={x},
  year={xx}
}
```
