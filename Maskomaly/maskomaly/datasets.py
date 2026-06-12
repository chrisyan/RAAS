from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from typing import List, Optional, Tuple

class FishyScapesLaF(Dataset):
    def __init__(self, root_path):
        self.images_paths = sorted(os.listdir(os.path.join(root_path, "original")))
        self.images_paths = [os.path.join(root_path, "original", image) for image in self.images_paths]

        self.labels_paths = sorted(os.listdir(os.path.join(root_path, "labels_masks")))
        self.labels_paths = [os.path.join(root_path, "labels_masks", label) for label in self.labels_paths]

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = cv2.imread(image_path)

        label_path = self.labels_paths[idx]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # read label in grayscale

        anomaly_gt = (label == 1).astype(np.uint8)     # 1 = anomaly
        ignore = (label == 255).astype(np.uint8)       # 255 = void

        return image, anomaly_gt, ignore, os.path.basename(image_path)

class FishyScapesStatic(Dataset):
    def __init__(self, root_path):
        self.images_paths = os.listdir(os.path.join(root_path, "images"))
        self.images_paths.sort()
        self.images_paths = [os.path.join(root_path, "images", image) for image in self.images_paths]

        self.labels_paths = os.listdir(os.path.join(root_path, "labels_masks"))
        self.labels_paths.sort()
        self.labels_paths = [os.path.join(root_path, "labels_masks", label) for label in self.labels_paths]

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = cv2.imread(image_path)

        label_path = self.labels_paths[idx]
        label = cv2.imread(label_path)
        label = label[:, :, 0]

        anomaly_gt = np.zeros_like(label, dtype=np.uint8)
        anomaly_gt[label == 1] = 1  # 1 = anomaly

        ignore = np.zeros_like(label, dtype=np.uint8)
        ignore[label == 255] = 1  # white is void, ignored

        return image, anomaly_gt, ignore, os.path.basename(image_path)


class SMIYCANO(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path

        # collect validation image paths
        self.images_paths = os.listdir(os.path.join(root_path, "images_val"))
        self.images_paths = [img for img in self.images_paths if img.startswith("validation")]
        self.images_paths.sort()

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_name = self.images_paths[idx]
        image_path = os.path.join(self.root_path, "images_val", image_name)

        label_base = image_name
        if label_base.endswith(".jpg") or label_base.endswith(".png"):
            label_base = label_base.rsplit(".", 1)[0]
        label_base = label_base + "_labels_semantic_color.png"
        label_path = os.path.join(self.root_path, "labels_masks", label_base)

        print("Image Path:", image_path)
        print("Label Path:", label_path)

        image = cv2.imread(image_path)
        print("image shape:", image.shape)

        label = cv2.imread(label_path)
        label = label[:, :, 1]  # use green channel for anomaly label
        print("label shape:", label.shape)

        anomaly_gt = np.zeros_like(label, dtype=np.uint8)
        anomaly_gt[label == 102] = 1
        print("anomaly_gt unique values:", np.unique(anomaly_gt))

        ignore = np.zeros_like(label, dtype=np.uint8)
        ignore[label == 0] = 1
        print("ignore unique values:", np.unique(ignore))

        return image, anomaly_gt, ignore, os.path.basename(image_path)


class SMIYCOBS(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path

        # collect validation image paths
        self.images_paths = os.listdir(os.path.join(root_path, "images_val"))
        self.images_paths = [img for img in self.images_paths if img.startswith("validation")]
        self.images_paths.sort()

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_name = self.images_paths[idx]
        image_path = os.path.join(self.root_path, "images_val", image_name)

        # derive grayscale label path from image name
        label_name = image_name.replace(".png", "_labels_semantic.png")
        label_path = os.path.join(self.root_path, "labels_masks", label_name)

        print("Image Path:", image_path)
        print("Label Path:", label_path)

        # read RGB image
        image = cv2.imread(image_path)
        print("image shape:", image.shape)

        # read grayscale label
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        print("label shape:", label.shape)
        print("label unique values:", np.unique(label, return_counts=True))

        # anomaly ground truth: 1 = anomaly
        anomaly_gt = np.zeros_like(label, dtype=np.uint8)
        anomaly_gt[label == 1] = 1

        # ignore mask: 1 = void
        ignore = np.zeros_like(label, dtype=np.uint8)
        ignore[label == 255] = 1

        print("anomaly_gt unique values:", np.unique(anomaly_gt))
        print("ignore unique values:", np.unique(ignore))

        return image, anomaly_gt, ignore, os.path.basename(image_path)

class RoadAnomaly(Dataset):
    def __init__(self, root_path):
        self.images_root = os.path.join(root_path, "original")
        self.labels_root = os.path.join(root_path, "labels")

        self.images_paths = sorted([
            os.path.join(self.images_root, f)
            for f in os.listdir(self.images_root)
            if f.endswith(".jpg")
        ])

        self.labels_paths = []
        for img_path in self.images_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.labels_root, base_name + ".png")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found for {img_path}")
            self.labels_paths.append(label_path)

        assert len(self.images_paths) == len(self.labels_paths), \
            "Number of images and labels do not match."

        print(f"Loaded {len(self.images_paths)} images and labels.")

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        label_path = self.labels_paths[idx]

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # single channel

        # anomaly ground truth: label == 1 is anomaly
        anomaly_gt = (label == 1).astype(np.uint8)

        # no void regions in this dataset
        ignore = np.zeros_like(label, dtype=np.uint8)

        filename = os.path.basename(image_path)

        return image, anomaly_gt, ignore, filename
