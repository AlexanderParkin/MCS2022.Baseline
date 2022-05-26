import os
import cv2
import pandas as pd
import torch.utils.data as data
from PIL import Image


def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_file))
    return img


class CustomDataset(data.Dataset):
    def __init__(self, root, annotation_file, transform):
        self.root = root
        self.transform = transform
        self.imlist = pd.read_csv(annotation_file).values.tolist()

    def __getitem__(self, index):
        impath, x1, y1, x2, y2, target = self.imlist[index]
        full_imname = os.path.join(self.root, impath)

        if not os.path.exists(full_imname):
            print('No file ', full_imname)

        img = read_image(full_imname)

        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))
        if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0 and x2 < img.shape[1] and y2 < img.shape[0] \
                and x1 < x2 and y1 < y2:
            img = img[y1: y2, x1: x2]

        img = Image.fromarray(img)
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imlist)
