"""1. inside road polygon
2. ID list
3. road query mask filter"""
import os
import sys
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine.defaults import DefaultPredictor

sys.path.append('/home/zhiranworkstation/raas/Mask2Former')
from mask2former import add_maskformer2_config

import clip
from PIL import Image
import numpy as np
import torch
import os

def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

# Cityscapes ID-to-color mapping (only main classes for brevity)
CITYSCAPES_COLORMAP = {
    0: (128, 64,128),   # road
    1: (244, 35,232),   # sidewalk
    2: ( 70, 70, 70),   # building
    3: (102,102,156),   # wall
    4: (190,153,153),   # fence
    5: (153,153,153),   # pole
    6: (250,170, 30),   # traffic light
    7: (220,220,  0),   # traffic sign
    8: (107,142, 35),   # vegetation
    9: (152,251,152),   # terrain
    10:( 70,130,180),   # sky
    11:(220, 20, 60),   # person
    12:(255,  0,  0),   # rider
    13:(  0,  0,142),   # car
    14:(  0,  0, 70),   # truck
    15:(  0, 60,100),   # bus
    16:(  0, 80,100),   # train
    17:(  0,  0,230),   # motorcycle
    18:(119, 11, 32),   # bicycle
    19:(  0,  0,  0),   # void/unlabeled
}


def colorize_segmentation(segmentation):
    h, w = segmentation.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CITYSCAPES_COLORMAP.items():
        color_mask[segmentation == class_id] = color
    return color_mask


class BaseModel():
    def __init__(self, args):
        cfg = setup_cfg(args)
        self.model = DefaultPredictor(cfg)
        self.times = []

    def get_soft_mask(self, image):
        raise Exception("Needs to be overloaded!")

    def get_time(self):
        return sum(self.times) / len(self.times)


class BaseSegmentationModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

    def get_probs_and_seg(self, image):
        segmentation, mask_cls_result, mask_pred_result = self.model(image)

        # print("segmentation: ", segmentation)

        print(f"mask_cls_result shape: {mask_cls_result.shape}")
        print(f"mask_pred_result shape: {mask_pred_result.shape}")

        mask_cls_result = F.softmax(mask_cls_result, dim=1).cpu().numpy()
        mask_pred_result = mask_pred_result.sigmoid().cpu().numpy()
        print(f"mask_cls_result shape- after: {mask_cls_result.shape}")
        print(f"mask_pred_result shape- after: {mask_pred_result.shape}")

        return mask_cls_result, mask_pred_result, segmentation



class Maskomaly(BaseSegmentationModel):
    def __init__(self, args):
        super().__init__(args)
        if args.analysis_file:
            self.cp = np.load(args.analysis_file)["cp"]
            self.ranking = np.argsort(self.cp)[::-1]
            self.cp = self.cp[self.ranking]
            self.take = int(args.masks) | np.argmax(self.cp < 0.25)
            self.ranking = self.ranking[:self.take]
        else:
            self.ranking = [49, 31, 83, 32]

        self.class_stats = np.zeros(10)
        self.pred_stats = np.zeros(10)

    def get_soft_mask(self, image, anomaly_path=None, output_base_dir=None, filename=None):
        original_size = image.shape[:2]  # (H, W)
        mask_cls_result, mask_pred_result, segmentation = self.get_probs_and_seg(image)
        # Convert the segmentation logits into a [H, W] class map 
        segmentation = segmentation["sem_seg"]
        segmentation = (segmentation.argmax(0).cpu().numpy())  # shape: [H, W]
        print("segmentation shape: ", segmentation.shape)
        seg_min = np.min(segmentation)
        seg_max = np.max(segmentation)
        print("seg_min: ", seg_min)
        print("seg_max: ", seg_max)
        print("########## maskomaly-ID ################")

        start_t = time.time()

        soft_mask = np.ones_like(mask_pred_result[0], dtype=np.float32)
        soft_mask2 = np.zeros_like(mask_pred_result[0], dtype=np.float32)

        for ind in self.ranking:
            conf = np.max(mask_cls_result[ind])
            self.class_stats[int(conf * 10)] += 1
            for t in range(10):
                self.pred_stats[t] += np.count_nonzero(mask_pred_result[ind] > t / 10)
            soft_mask2 = np.maximum(soft_mask2, mask_pred_result[ind] * conf)

        for i in range(mask_cls_result.shape[0]):
            max_cls = np.argmax(mask_cls_result[i])
            conf = mask_cls_result[i][max_cls]
            if max_cls != 19 and conf > 0.7:
                for t in range(10):
                    self.pred_stats[t] += np.count_nonzero(mask_pred_result[i] > t / 10)
                self.class_stats[int(conf * 10)] += 1
                neg = 1 - mask_pred_result[i] * conf
                soft_mask = np.minimum(soft_mask, neg)

        max_indices = np.argmax(mask_cls_result, axis=1)
        positive = mask_pred_result[max_indices != 19.0]
        for i in range(positive.shape[0]):
            for j in range(i + 1, positive.shape[0]):
                overlap = np.logical_and(positive[i] > 0.1, positive[j] > 0.1)
                soft_mask = np.minimum(soft_mask, 1 - overlap.astype(np.float32))

        for i in [19, 24]:
            conf = np.max(mask_cls_result[i])
            neg = 1 - mask_pred_result[i] * conf
            soft_mask = np.minimum(soft_mask, neg)

        soft_mask = 0.6 * soft_mask + 0.4 * soft_mask2
        soft_mask = cv2.resize(soft_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_AREA)

        H, W = image.shape[:2]

        #  Road polygon + CLIP filtering (always runs) 
        road_index = 20
        road_query_mask_binary = (mask_pred_result[road_index] > 0.5).astype(np.uint8) * 255
        road_query_mask_binary_resized = cv2.resize(road_query_mask_binary, (W, H), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(road_query_mask_binary_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        road_polygon_mask = np.zeros_like(road_query_mask_binary_resized)
        cv2.fillPoly(road_polygon_mask, contours, 255)

        initial_road_anomaly_mask = np.logical_and(
            (road_polygon_mask == 255),
            (road_query_mask_binary_resized == 0)
        ).astype(np.float32)

        #  Save debug files only when output_base_dir is provided 
        if output_base_dir and filename:
            cv2.imwrite(f"{output_base_dir}/{filename}_road_query_mask_binary.png",
                        road_query_mask_binary_resized)
            cv2.imwrite(f"{output_base_dir}/{filename}_initial_road_anomaly_mask.png",
                        (initial_road_anomaly_mask * 255).astype('uint8'))
            anomaly_patch_dir = os.path.join(output_base_dir, "anomaly_patches")
            os.makedirs(anomaly_patch_dir, exist_ok=True)
        else:
            anomaly_patch_dir = None

        if isinstance(image, torch.Tensor):
            rgb_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        elif isinstance(image, np.ndarray):
            rgb_image = image.copy()
        else:
            raise TypeError("Unsupported image format for RGB extraction")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)

        road_anomaly_mask_uint8 = (initial_road_anomaly_mask * 255).astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(road_anomaly_mask_uint8)

        id_prompts = [
            "a photo of road",
            "a photo of sidewalk",
            "a photo of building",
            "a photo of wall",
            "a photo of fence",
            "a photo of pole",
            "a photo of traffic light",
            "a photo of traffic sign",
            "a photo of vegetation",
            "a photo of terrain",
            "a photo of sky",
            "a photo of person on the road",
            "a photo of rider on the road",
            "a photo of car on the road",
            "a photo of truck on the road",
            "a photo of bus on the road",
            "a photo of train on the track",
            "a photo of motorcycle on the road",
            "a photo of bicycle on the road"
        ]
        text = clip.tokenize(id_prompts).to(device)

        for label in range(1, num_labels):
            component_mask = (labels_im == label).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(component_mask)

            if w <= 1 or h <= 1:
                print(f"[Warning] Skipping small patch: {w}x{h}")
                continue

            patch = rgb_image[y:y + h, x:x + w]
            if patch.size == 0:
                print(f"[Warning] Empty patch at label {label}")
                continue

            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            if anomaly_patch_dir:
                cv2.imwrite(os.path.join(anomaly_patch_dir, f"{filename}_patch_{label:03d}.png"), patch)

            patch_pil = Image.fromarray(patch)
            image_patch = preprocess(patch_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                logits_per_image, _ = clip_model(image_patch, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten()

            max_id_prob = np.max(probs)
            predicted_class = id_prompts[np.argmax(probs)]

            if max_id_prob > 0.85:
                print(f"[CLIP] Patch {label} | ID: {predicted_class} ({max_id_prob:.2f}) → skip")
                soft_mask[y:y + h, x:x + w][component_mask[y:y + h, x:x + w] > 0] = 0.05
            else:
                print(f"[CLIP] Patch {label} | Unknown ({max_id_prob:.2f}) → anomaly")
                soft_mask[y:y + h, x:x + w][component_mask[y:y + h, x:x + w] > 0] = 1.0

        self.times.append(time.time() - start_t)

        return soft_mask

