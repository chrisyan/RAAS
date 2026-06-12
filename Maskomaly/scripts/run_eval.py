"""End-to-end OOD inference + evaluation in a single pass.

No intermediate .npz files required. Results are accumulated in memory and
metrics are computed at the end.

Usage:
    python run_eval.py --model maskomaly --dataset fs_laf \
        --input /path/to/dataset \
        --output /path/to/results

    python run_eval.py --model maskomaly_id --dataset smiyc_anomaly \
        --input xxx \
        --output xxx \
        --debug

    python run_eval.py --model maskomaly_ood --dataset fs_static \
        --input /path/to/fs_static \
        --output /path/to/results \
        --no_heatmaps

    # --input and --output are optional; defaults are used when omitted
    python run_eval.py --model maskomaly --dataset roadanomaly

Models  : maskomaly | maskomaly_id | maskomaly_ood
Datasets: fs_laf | fs_static | smiyc_anomaly | smiyc_obstacle | roadanomaly
"""

import argparse
import importlib
import inspect
import multiprocessing as mp
import os
import sys
import time
import traceback

import cv2
import numpy as np
import tqdm
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

# ── path setup, please use custom paths ────────────────────────────────────────────────────────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
_MASKOMALY_DIR = os.path.join(_PROJECT_DIR, "maskomaly")
_DETECTRON2_REPLACEMENTS_DIR = os.path.join(_PROJECT_DIR, "detectron2_replacements")
_DETECTRON2_DIR = "/home/zhiranworkstation/raas/detectron2"

for _p in [_DETECTRON2_REPLACEMENTS_DIR, _MASKOMALY_DIR, _DETECTRON2_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from eval import calculate_auroc, calculate_aupr, get_scores

# ── constants, please use custom paths ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = (
    "/home/zhiranworkstation/raas/Mask2Former/configs/"
    "cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml"
)
DEFAULT_WEIGHTS = (
    "/home/zhiranworkstation/raas/Maskomaly/maskomaly/ckpt/model_final_17c1ee.pkl"
)
DEFAULT_OUTPUT_BASE = (
    "/media/zhiranworkstation/Expansion/1_gaiax/IV2026_anomaly_segmentation"
)

MODEL_MODULES = {
    "maskomaly":     "model_ori",
    "maskomaly_id":  "model_id",
    "maskomaly_ood": "model_ood",
}

DATASET_CONFIGS = {
    "fs_laf": (
        "FishyScapesLaF",
        "/media/zhiranworkstation/T7a/datasets/Anomaly_dataset/fishyscapes/LostAndFound",
    ),
    "fs_static": (
        "FishyScapesStatic",
        "/home/zhiranworkstation/anomaly_detection/mask2anomaly/Validation_Dataset/fs_static",
    ),
    "roadanomaly": (
        "RoadAnomaly",
        "/media/zhiranworkstation/T7a/datasets/Anomaly_dataset/fishyscapes/road_anomaly",
    ),
    "smiyc_anomaly": (
        "SMIYCANO",
        "/media/zhiranworkstation/T7a/datasets/Anomaly_dataset/SMIYC/dataset_AnomalyTrack",
    ),
    "smiyc_obstacle": (
        "SMIYCOBS",
        "/media/zhiranworkstation/T7a/datasets/Anomaly_dataset/SMIYC/dataset_ObstacleTrack",
    ),
}


# ── argument parsing ──────────────────────────────────────────────────────────

def get_parser():
    parser = argparse.ArgumentParser(
        description="Unified OOD inference + evaluation for Maskomaly",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- model / dataset ---
    parser.add_argument("--model", choices=list(MODEL_MODULES), required=True)
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS), required=True)
    parser.add_argument("--input", default=None,
                        help="Dataset root. Defaults to standard path for the dataset.")

    # --- output ---
    parser.add_argument("--output", default=None,
                        help="Directory for results.txt and optional heatmaps/npz. "
                             "Defaults to <output_base>/<model>/<dataset>.")
    parser.add_argument("--output_base", default=DEFAULT_OUTPUT_BASE)

    # --- debug ---
    parser.add_argument("--debug", action="store_true",
                        help="Save road-filter debug images (road masks, anomaly patches).")
    parser.add_argument("--output_debug", default=None,
                        help="Directory for debug output. Defaults to <output>_debug.")

    # --- evaluation ---
    parser.add_argument("--strategy", choices=["accumulate", "per_image"],
                        default="accumulate",
                        help="'accumulate': stack all images then score globally. "
                             "'per_image': score each image then average.")
    parser.add_argument("--no_heatmaps", action="store_true",
                        help="Skip saving jet-colormap heatmap overlays.")

    # --- optional npz saving ---
    parser.add_argument("--save_npz", action="store_true",
                        help="Also save per-image .npz files (image/gt/ignore/soft_mask).")

    # --- model hyperparams ---
    parser.add_argument("--masks", type=int, default=4)
    parser.add_argument("--analysis_file", default=None)
    parser.add_argument("--config-file", metavar="FILE", default=DEFAULT_CONFIG)
    parser.add_argument("--opts", nargs=argparse.REMAINDER,
                        default=["MODEL.WEIGHTS", DEFAULT_WEIGHTS])
    return parser


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(model_alias, args):
    module_name = MODEL_MODULES[model_alias]
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import '{module_name}'. Check {_MASKOMALY_DIR}.\n{e}"
        )
    return module.Maskomaly(args)


def load_dataset(dataset_alias, input_path):
    class_name, default_path = DATASET_CONFIGS[dataset_alias]
    root = input_path or default_path
    import datasets as ds_module
    return getattr(ds_module, class_name)(root)


def supports_debug_kwargs(model):
    return "output_base_dir" in inspect.signature(model.get_soft_mask).parameters


def call_get_soft_mask(model, image, output_debug, filename, debug):
    if supports_debug_kwargs(model):
        base = output_debug if debug else None
        result = model.get_soft_mask(
            image,
            output_base_dir=base,
            filename=filename if debug else None,
        )
    else:
        result = model.get_soft_mask(image)
    return result[0] if isinstance(result, tuple) else result


def resize_to_gt(soft_mask, gt):
    if soft_mask.shape != gt.shape:
        soft_mask = cv2.resize(soft_mask, gt.shape[::-1], interpolation=cv2.INTER_CUBIC)
    return soft_mask


def make_heatmap(image, probs):
    hm = cv2.applyColorMap((probs * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(hm, 0.5, image, 0.5, 0)


def write_results(output_dir, ap, roc, fpr95, aupr):
    path = os.path.join(output_dir, "results.txt")
    with open(path, "w") as f:
        f.write(f"AP      : {100 * ap:.3f}%\n")
        f.write(f"AUROC   : {100 * roc:.2f}%\n")
        f.write(f"FPR@95  : {100 * fpr95:.2f}%\n")
        f.write(f"AUPR    : {100 * aupr:.2f}%\n")
    print(f"\n{'=' * 40}")
    print(f"AP      : {100 * ap:.3f}%")
    print(f"AUROC   : {100 * roc:.2f}%")
    print(f"FPR@95  : {100 * fpr95:.2f}%")
    print(f"AUPR    : {100 * aupr:.2f}%")
    print(f"Results : {path}")


# ── evaluation helpers ────────────────────────────────────────────────────────

def score_per_image(gt, ignore, soft_mask):
    """Compute metrics for a single image. Returns None if image is degenerate."""
    valid = ignore.flatten() < 1
    gt_v = gt.flatten()[valid]
    pred_v = soft_mask.flatten()[valid]
    if len(np.unique(gt_v)) < 2:
        return None
    ap = average_precision_score(gt_v, pred_v)
    roc, fpr95, _ = calculate_auroc(gt_v, pred_v)
    aupr = calculate_aupr(gt_v, pred_v)
    return ap, roc, fpr95, aupr


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    output_tag = f"output_{args.model}"
    output_dir = args.output or os.path.join(args.output_base, output_tag, args.dataset)
    output_debug = args.output_debug or (output_dir + "_debug")

    os.makedirs(output_dir, exist_ok=True)
    if args.debug:
        os.makedirs(output_debug, exist_ok=True)

    save_heatmaps = not args.no_heatmaps

    print(f"Model    : {args.model}  ({MODEL_MODULES[args.model]})")
    print(f"Dataset  : {args.dataset}  ({DATASET_CONFIGS[args.dataset][0]})")
    print(f"Input    : {args.input or DATASET_CONFIGS[args.dataset][1]}")
    print(f"Output   : {output_dir}")
    print(f"Strategy : {args.strategy}")
    if args.debug:
        print(f"Debug    : {output_debug}")
    if args.save_npz:
        print(f"NPZ      : saving to {output_dir}")

    dataset = load_dataset(args.dataset, args.input)
    dataloader = DataLoader(dataset)
    model = load_model(args.model, args)

    # accumulators for 'accumulate' strategy
    gt_list, ignore_list, mask_list = [], [], []
    # accumulators for 'per_image' strategy
    ap_vals, roc_vals, fpr_vals, aupr_vals = [], [], [], []
    skipped = 0

    t0 = time.time()

    for idx, (image, gt, ignore, file) in enumerate(tqdm.tqdm(dataloader, desc="Inference")):
        try:
            image = image.cpu().numpy()[0]
            gt = gt.cpu().numpy()[0]
            ignore = ignore.cpu().numpy()[0]
            filename = os.path.splitext(str(file[0]))[0]

            soft_mask = call_get_soft_mask(model, image, output_debug, filename, args.debug)
            soft_mask = resize_to_gt(soft_mask, gt)

            print(f"{filename} | soft_mask [{soft_mask.min():.4f}, {soft_mask.max():.4f}]"
                  f" | mean {soft_mask.mean():.4f}")

            if save_heatmaps:
                hm = make_heatmap(image.copy(), soft_mask)
                cv2.imwrite(os.path.join(output_dir, f"{idx:04d}_heat.png"), hm)

            if args.save_npz:
                np.savez(
                    os.path.join(output_dir, f"{filename}.npz"),
                    image=image, gt=gt, ignore=ignore, soft_mask=soft_mask,
                )

            if args.strategy == "accumulate":
                gt_list.append(gt)
                ignore_list.append(ignore)
                mask_list.append(soft_mask)
            else:
                scores = score_per_image(gt, ignore, soft_mask)
                if scores is None:
                    print(f"  Single-class image, skipping metrics: {filename}")
                    skipped += 1
                else:
                    ap_vals.append(scores[0])
                    roc_vals.append(scores[1])
                    fpr_vals.append(scores[2])
                    aupr_vals.append(scores[3])

        except Exception:
            print(f"[Error] Failed on: {file[0]}")
            traceback.print_exc()
            continue

    # ── compute final metrics ─────────────────────────────────────────────────
    if args.strategy == "accumulate":
        gt_arr = np.asarray(gt_list)
        ignore_arr = np.asarray(ignore_list)
        mask_arr = np.asarray(mask_list)
        ap, roc, fpr95, aupr = get_scores(gt_arr, mask_arr, ignore_arr, mode="total")
    else:
        if skipped:
            print(f"\nSkipped {skipped} single-class images.")
        ap = float(np.nanmean(ap_vals))
        roc = float(np.nanmean(roc_vals))
        fpr95 = float(np.nanmean(fpr_vals))
        aupr = float(np.nanmean(aupr_vals))

    write_results(output_dir, ap, roc, fpr95, aupr)
    print(f"Total time : {time.time() - t0:.1f}s")

    if model.times:
        print(f"Avg infer  : {sum(model.times) / len(model.times):.4f}s/image")


if __name__ == "__main__":
    main()
