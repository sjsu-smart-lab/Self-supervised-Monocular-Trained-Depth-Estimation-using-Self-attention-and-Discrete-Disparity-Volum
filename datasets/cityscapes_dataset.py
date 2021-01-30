import os
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path


class CityscapesData(Dataset):
    """
    Cityscapes dataset for inference
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.all_imgs = sorted(list(Path(folder_path).glob('**/*.png')))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        image_path = self.all_imgs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        # resize in accordance to the trained model and normalize
        image = (cv2.resize(image, (640, 192), cv2.INTER_AREA)*1.0)/256
        image = image.astype(np.float32)

        return image