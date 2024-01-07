import os
import random
import numpy as np
from tqdm import tqdm
import cv2
import glob

START_TRAIN_FOLDER = 1
END_TRAIN_FOLDER = 56

def save (name , split):
    img = cv2.imread(f"AllImages/Image/{name}.png")
    vessel = cv2.imread(f"AllImages/Vessel/{name}.png")
    mask = cv2.imread(f"AllImages/Sclera/{name}.png")

    cv2.imwrite(f"Data/FullImages/{split}/image_org/{name}.png",img)
    cv2.imwrite(f"Data/FullImages/{split}/mask/{name}.png",vessel)
    cv2.imwrite(f"Data/FullImages/{split}/mask_sclera/{name}.png",mask)


if __name__ == "__main__":
    # Extract all labeled images (128)
    DATA_PATH = 'SBVPI-org/'
    for n in tqdm(range(START_TRAIN_FOLDER, END_TRAIN_FOLDER), total=END_TRAIN_FOLDER - START_TRAIN_FOLDER):
        path = DATA_PATH + str(n) + '/'
        files = next(os.walk(path))[2]
        for file in files:
            if os.path.join(path, file).endswith("_vessels.png"):
                vessel = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
                img = cv2.imread(os.path.join(path, file[:-12] + '.jpg'))
                sclera = cv2.imread(os.path.join(path, file[:-12] + '_sclera.png'), cv2.IMREAD_GRAYSCALE)

                cv2.imwrite(f"AllImages/Image/{file[:-12]}.png" ,img)
                cv2.imwrite(f"AllImages/Sclera/{file[:-12]}.png" ,sclera)
                cv2.imwrite(f"AllImages/Vessel/{file[:-12]}.png" ,vessel)

    # split data into -> train-test-valid
    NEW_PATH = "AllImages/"
    images_paths = sorted(glob.glob(os.path.join(NEW_PATH, "Image", "*.png")))
    random.shuffle(images_paths)
    
    for idx in tqdm(range(0,len(images_paths))):
        name = images_paths[idx].split("\\")[-1].split(".")[0]
        if idx <= 100:
            save(name , "train")
        if idx > 100 and idx < 114:
            save(name , "valid")
        if idx > 114:
            save(name , "test")