"""Inference on a folder of images — no ground-truth required.

Saves soft masks and heatmap overlays. Optionally saves query masks (maskomaly
only) and road-filter debug images (maskomaly_id / maskomaly_ood with --debug).

Usage:
    python infer.py --model maskomaly \
        --input /path/to/images \
        --output /path/to/results

    python infer.py --model maskomaly_id \
        --input /path/to/images \
        --output /path/to/results \
        --debug

    python infer.py --model maskomaly_ood \
        --input /path/to/images \
        --output /path/to/results \
        --ext jpg

Models: maskomaly | maskomaly_id | maskomaly_ood
"""

import argparse
import importlib
import inspect
import os
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
_MASKOMALY_DIR = os.path.join(_PROJECT_DIR, "maskomaly")
_DETECTRON2_REPLACEMENTS_DIR = os.path.join(_PROJECT_DIR, "detectron2_replacements")
_DETECTRON2_DIR = "/home/zhiranworkstation/raas/detectron2"

for _p in [_DETECTRON2_REPLACEMENTS_DIR, _MASKOMALY_DIR, _DETECTRON2_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = (
    "/home/zhiranworkstation/raas/Mask2Former/configs/"
    "cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml"
)
DEFAULT_WEIGHTS = (
    "/home/zhiranworkstation/raas/Maskomaly/maskomaly/ckpt/model_final_17c1ee.pkl"
)

MODEL_MODULES = {
    "maskomaly":     "model_ori",
    "maskomaly_id":  "model_id",
    "maskomaly_ood": "model_ood",
}


# ── argument parsing ──────────────────────────────────────────────────────────

def get_parser():
    parser = argparse.ArgumentParser(
        description="Maskomaly inference on a folder of images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=list(MODEL_MODULES), required=True,
                        help="Model variant.")
    parser.add_argument("--input", required=True,
                        help="Directory containing input images.")
    parser.add_argument("--output", required=True,
                        help="Directory to save results.")
    parser.add_argument("--ext", default="",
                        help="Image extension to filter by (png/jpg). "
                             "Leave empty to pick up png, jpg, and jpeg automatically.")
    parser.add_argument("--debug", action="store_true",
                        help="Save road-filter debug images (maskomaly_id/ood only).")
    parser.add_argument("--masks", type=int, default=4)
    parser.add_argument("--analysis_file", default=None)
    parser.add_argument("--config-file", metavar="FILE", default=DEFAULT_CONFIG)
    parser.add_argument("--opts", nargs=argparse.REMAINDER,
                        default=["MODEL.WEIGHTS", DEFAULT_WEIGHTS])
    return parser


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(model_alias, args):
    module = importlib.import_module(MODEL_MODULES[model_alias])
    return module.Maskomaly(args)


def supports_debug_kwargs(model):
    return "output_base_dir" in inspect.signature(model.get_soft_mask).parameters


def run_inference(model, image, output_debug, stem, debug):
    """Call get_soft_mask with the right signature and unpack the result."""
    if supports_debug_kwargs(model):
        base = output_debug if debug else None
        result = model.get_soft_mask(
            image,
            output_base_dir=base,
            filename=stem if debug else None,
        )
        soft_mask = result[0] if isinstance(result, tuple) else result
        query_masks = None
    else:
        result = model.get_soft_mask(image)
        if isinstance(result, tuple):
            soft_mask, query_masks = result[0], result[1]
        else:
            soft_mask, query_masks = result, None
    return soft_mask, query_masks


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_parser().parse_args()

    softmask_dir = os.path.join(args.output, "soft_mask")
    heatmap_dir  = os.path.join(args.output, "heatmap")
    debug_dir    = os.path.join(args.output, "debug")
    query_dir    = os.path.join(args.output, "query_masks")

    os.makedirs(softmask_dir, exist_ok=True)
    os.makedirs(heatmap_dir,  exist_ok=True)
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)

    print(f"Model  : {args.model}  ({MODEL_MODULES[args.model]})")
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    if args.debug:
        print(f"Debug  : {debug_dir}")

    model = load_model(args.model, args)

    if args.ext:
        exts = {args.ext.lstrip(".").lower()}
    else:
        exts = {"png", "jpg", "jpeg"}

    image_files = sorted(
        f for f in os.listdir(args.input)
        if f.rsplit(".", 1)[-1].lower() in exts
    )
    if not image_files:
        print(f"[Error] No images ({', '.join(exts)}) found in {args.input}")
        return

    print(f"Images : {len(image_files)} found")

    times_ms = []

    for filename in tqdm(image_files):
        image_path = os.path.join(args.input, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"[Warning] Could not load: {image_path}")
            continue

        stem = os.path.splitext(filename)[0]

        t0 = time.perf_counter()
        soft_mask, query_masks = run_inference(model, image, debug_dir, stem, args.debug)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)

        print(f"[{filename}] min={soft_mask.min():.4f} max={soft_mask.max():.4f} "
              f"mean={soft_mask.mean():.4f} time={elapsed_ms:.1f}ms")

        # soft mask (grayscale)
        soft_mask_8bit = (soft_mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(softmask_dir, f"{stem}_soft_mask.png"), soft_mask_8bit)

        # heatmap overlay
        heatmap = cv2.applyColorMap(soft_mask_8bit, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(os.path.join(heatmap_dir, f"{stem}_heatmap.png"), blended)

        # query masks (maskomaly only — returns soft_mask_queries)
        if query_masks is not None:
            qdir = os.path.join(query_dir, stem)
            os.makedirs(qdir, exist_ok=True)
            for i in range(query_masks.shape[0]):
                qm = (query_masks[i] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(qdir, f"query_{i:03d}.png"), qm)

    if times_ms:
        print(f"\nAverage inference time: {sum(times_ms) / len(times_ms):.1f} ms/image")


if __name__ == "__main__":
    main()
