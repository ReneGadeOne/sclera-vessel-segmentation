import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion, CoarseDropout

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    """ X = Images and Y = masks """
    train_x = sorted(glob(os.path.join(path, "train", "image", "*.png")))
    train_y = sorted(glob(os.path.join(path, "train", "labels", "*.png")))
     
    return (train_x, train_y) 

def augment_data(images, masks, save_path, augment=True):
    H = 256
    W = 256

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting names """
        name = x.split("\\")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            X = [x, x1, x3, x4]
            Y = [y, y1, y3, y4]


        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            # i = cv2.resize(i, (W, H))
            # m = cv2.resize(m, (W, H))

            if len(X) == 1:
                tmp_image_name = f"{name}.png"
                tmp_mask_name = f"{name}.png"
            else:
                tmp_image_name = f"{name}_{index}.png"
                tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "labels", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "Data/Crops"
    (train_x, train_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
   

    augment_data(train_x, train_y, "Data/Crops/train/", augment=True)
