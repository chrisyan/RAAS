================================================================================
  MASKOMALY — Anomaly Segmentation with Mask2Former
================================================================================

Three model variants:
  maskomaly      Original Maskomaly (hardcoded mask indices, no CLIP)
  maskomaly_id   Road-aware filtering + CLIP, ID prompts only (19 Cityscapes)
  maskomaly_ood  Road-aware filtering + CLIP, ID + OOD prompts

Five benchmark datasets:
  fs_laf         Fishyscapes LostAndFound
  fs_static      Fishyscapes Static
  smiyc_anomaly  SMIYC AnomalyTrack
  smiyc_obstacle SMIYC ObstacleTrack
  roadanomaly    RoadAnomaly21


================================================================================
  1. REPOSITORY LAYOUT
================================================================================

  raas/
  ├── detectron2/          <- detectron2 source (must be cloned here)
  ├── Mask2Former/         <- Mask2Former source (must be cloned here)
  └── Maskomaly/           <- this repo
      ├── maskomaly/
      │   ├── ckpt/        <- place model_final_17c1ee.pkl here
      │   ├── model_ori.py <- maskomaly
      │   ├── model_id.py  <- maskomaly_id
      │   ├── model_ood.py <- maskomaly_ood
      │   └── datasets.py
      ├── detectron2_replacements/   <- patched DefaultPredictor (returns 3 values)
      └── scripts/
          ├── run_eval.py  <- inference + evaluation in one pass
          └── infer.py     <- inference on an image folder (no ground truth needed)


================================================================================
  2. ENVIRONMENT SETUP
================================================================================

  conda env create -f environment.yml
  conda activate raas

  # Install OpenAI CLIP (required for maskomaly_id and maskomaly_ood)
  pip install git+https://github.com/openai/CLIP.git

  # Install Mask2Former dependencies
  cd /path/to/raas/Mask2Former
  pip install -r requirements.txt

  # Build detectron2 from source (do NOT pip install detectron2 separately)
  cd /path/to/raas/detectron2
  pip install -e .

  Note: the patched DefaultPredictor in detectron2_replacements/ must shadow
  the standard one. The scripts handle this via sys.path ordering — do not
  install a second detectron2 via pip.


================================================================================
  3. MODEL CHECKPOINT
================================================================================

  Download model_final_17c1ee.pkl and place it at:

    Maskomaly/maskomaly/ckpt/model_final_17c1ee.pkl

  This is the Mask2Former Swin-L backbone trained on Cityscapes.


================================================================================
  4. PATHS TO CONFIGURE
================================================================================

  Open scripts/run_eval.py and scripts/infer.py and update these constants
  near the top of each file:

  _DETECTRON2_DIR
      Path to the detectron2 source root.
      Default: /home/zhiranworkstation/raas/detectron2
      Change to: /your/path/raas/detectron2

  DEFAULT_CONFIG
      Path to the Mask2Former YAML config file.
      Default: .../Mask2Former/configs/cityscapes/semantic-segmentation/
                 swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml
      Change to match your Mask2Former clone location.

  DEFAULT_WEIGHTS
      Path to the model checkpoint .pkl file.
      Default: .../Maskomaly/maskomaly/ckpt/model_final_17c1ee.pkl
      Change to match your clone location.

  DEFAULT_OUTPUT_BASE  (run_eval.py only)
      Base directory where per-run result folders are created.
      Default: /media/zhiranworkstation/Expansion/1_gaiax/IV2026_anomaly_segmentation
      Change to any writable directory on your machine.

  DATASET_CONFIGS  (run_eval.py only)
      Default root path for each benchmark dataset.
      Change the paths inside this dict to match where you store the datasets:

        "fs_laf":         ("..", "/your/path/LostAndFound"),
        "fs_static":      ("..", "/your/path/fs_static"),
        "roadanomaly":    ("..", "/your/path/road_anomaly"),
        "smiyc_anomaly":  ("..", "/your/path/dataset_AnomalyTrack"),
        "smiyc_obstacle": ("..", "/your/path/dataset_ObstacleTrack"),

  All paths can also be overridden at runtime via CLI flags without editing the
  file (see sections 6 and 7).


================================================================================
  5. DATASET PREPARATION
================================================================================

  Expected folder structure for each dataset:

  Fishyscapes LostAndFound (fs_laf)
    <root>/
    ├── original/          <- RGB images (.png)
    └── labels/            <- label masks (0=background, 1=anomaly, 255=ignore)

  Fishyscapes Static (fs_static)
    <root>/
    ├── images/
    └── labels_masks/

  SMIYC AnomalyTrack (smiyc_anomaly)
    <root>/
    ├── images/            <- .jpg images
    └── labels_masks/      <- .png masks

  SMIYC ObstacleTrack (smiyc_obstacle)
    <root>/
    ├── images/
    └── labels_masks/

  RoadAnomaly21 (roadanomaly)
    <root>/
    ├── frames/
    └── frames_mask/

  To use a different path at runtime, pass --input <path> (see section 6).


================================================================================
  6. EVALUATION  (scripts/run_eval.py)
================================================================================

  Runs inference and computes AP / AUROC / FPR@95 / AUPR in a single pass.
  No intermediate .npz files are written by default.

  --- Activate environment ---

    conda activate raas
    cd /path/to/Maskomaly/scripts

  --- Basic usage (uses default paths from DATASET_CONFIGS) ---

    python run_eval.py --model maskomaly     --dataset fs_laf
    python run_eval.py --model maskomaly     --dataset fs_static
    python run_eval.py --model maskomaly     --dataset smiyc_anomaly
    python run_eval.py --model maskomaly     --dataset smiyc_obstacle
    python run_eval.py --model maskomaly     --dataset roadanomaly

    python run_eval.py --model maskomaly_id  --dataset smiyc_anomaly
    python run_eval.py --model maskomaly_ood --dataset smiyc_anomaly

  --- Specify paths explicitly ---

    python run_eval.py --model maskomaly_id --dataset fs_laf \
        --input  /path/to/LostAndFound \
        --output /path/to/results

  --- Save road-filter debug images (maskomaly_id / maskomaly_ood only) ---

    python run_eval.py --model maskomaly_id --dataset smiyc_anomaly --debug

  --- Per-image metric averaging instead of global accumulation ---

    python run_eval.py --model maskomaly_ood --dataset smiyc_obstacle \
        --strategy per_image

  --- Also save .npz files per image ---

    python run_eval.py --model maskomaly --dataset fs_laf --save_npz

  All flags:
    --model          maskomaly | maskomaly_id | maskomaly_ood
    --dataset        fs_laf | fs_static | smiyc_anomaly | smiyc_obstacle | roadanomaly
    --input          dataset root (overrides DATASET_CONFIGS default)
    --output         results directory (overrides DEFAULT_OUTPUT_BASE default)
    --output_base    base dir when --output is not set
    --strategy       accumulate (default) | per_image
    --debug          save road mask and CLIP patch debug images
    --output_debug   directory for debug images (default: <output>_debug)
    --no_heatmaps    skip heatmap saving (fastest, pure metrics)
    --save_npz       also write per-image .npz files to disk
    --config-file    path to Mask2Former YAML config (overrides DEFAULT_CONFIG)
    --opts           extra detectron2 key-value pairs, e.g.:
                       --opts MODEL.WEIGHTS /path/to/model.pkl

  Output structure:
    <output>/
    ├── results.txt        <- AP / AUROC / FPR@95 / AUPR
    └── XXXX_heat.png      <- heatmap overlays (unless --no_heatmaps)


================================================================================
  7. INFERENCE ON AN IMAGE FOLDER  (scripts/infer.py)
================================================================================

  No ground truth required. Saves soft masks and heatmap overlays.

  --- Basic usage ---

    conda activate raas
    cd /path/to/Maskomaly/scripts

    python infer.py --model maskomaly \
        --input  /path/to/images \
        --output /path/to/results

  --- With road-filter debug output ---

    python infer.py --model maskomaly_id \
        --input  /path/to/images \
        --output /path/to/results \
        --debug

  --- Filter by extension (auto-detects png/jpg/jpeg by default) ---

    python infer.py --model maskomaly_ood \
        --input  /path/to/images \
        --output /path/to/results \
        --ext jpg

  All flags:
    --model      maskomaly | maskomaly_id | maskomaly_ood
    --input      directory containing input images  [required]
    --output     directory to save results          [required]
    --ext        file extension filter (leave empty for auto png/jpg/jpeg)
    --debug      save road-filter debug images (maskomaly_id/ood only)
    --config-file / --opts   same as run_eval.py

  Output structure:
    <output>/
    ├── soft_mask/         <- grayscale anomaly score maps (*_soft_mask.png)
    ├── heatmap/           <- jet colormap blended on original (*_heatmap.png)
    ├── query_masks/       <- per-query masks (maskomaly only)
    └── debug/             <- road polygon / CLIP patch images (--debug only)


================================================================================
  8. MODEL DETAILS
================================================================================

  maskomaly (model_ori.py)
    Mask2Former with Swin-L backbone trained on Cityscapes.
    Anomaly score derived from low confidence across all Cityscapes classes.
    Returns (soft_mask, query_masks).

  maskomaly_id (model_id.py)
    Same backbone. Adds road-aware post-processing:
      1. Extracts road polygon from query mask #20.
      2. Finds regions inside the polygon not covered by the road mask.
      3. Classifies each region with CLIP against 19 Cityscapes ID prompts.
         - High ID confidence (> 0.85) -> suppress (score = 0.05)
         - Low ID confidence           -> mark as anomaly (score = 1.0)

  maskomaly_ood (model_ood.py)
    Same as maskomaly_id but uses both ID and OOD prompts together.
    OOD prompts: "something unusual in a driving scene", etc.
    Decision: if OOD prob dominates and ID prob < 0.85 -> anomaly.


================================================================================
  9. TROUBLESHOOTING
================================================================================

  AttributeError: module 'clip' has no attribute 'load'
    Wrong clip package installed. Fix:
      pip uninstall clip -y
      pip install git+https://github.com/openai/CLIP.git

  ValueError: not enough values to unpack (expected 3, got 1)
    The standard detectron2 DefaultPredictor is being used instead of the
    patched one in detectron2_replacements/. Make sure you did NOT pip install
    detectron2 separately. The scripts insert detectron2_replacements/ first
    in sys.path automatically.

  ImportError: _C.so undefined symbol
    ABI mismatch between the compiled detectron2 C extension and PyTorch.
    Rebuild detectron2 from source against the exact PyTorch version in the
    raas conda environment (PyTorch 1.9.0, CUDA 11.1).

  FileNotFoundError: labels_masks/
    Dataset folder structure does not match expected layout.
    Check section 5 and pass --input with the correct root path.

  No images found in <dir>
    Images may be .jpg while the extension filter is set to png.
    Pass --ext jpg or leave --ext empty for auto-detection (png/jpg/jpeg).
