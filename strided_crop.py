import cv2
import numpy as np
import os
import argparse
import glob
import tqdm


def strided_crop(img, label, height, width, name,stride=1):
    directories = [f'Data/Crops/{args.data}',f'Data/Crops/{args.data}/image',f'Data/Crops/{args.data}/labels']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    max_x = int(((img.shape[0]-height)/stride)+1)
    max_y = int(((img.shape[1]-width)/stride)+1)
    max_crops = (max_x)*(max_y)
    i = 0
    for h in range(max_x):
        for w in range(max_y):
                crop_img = img[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                crop_label = label[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                
                img_path = directories[1] + "/" + name  + str(i+1)+".png"                
                mask_path = directories[2] + "/" + name + str(i+1)+".png"
                
                cv2.imwrite(mask_path,crop_label)
                cv2.imwrite(img_path,crop_img)
                i = i + 1
               

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--data', type=str, default='train', required=False, choices=['train' , 'valid' , 'test'])
    args = parser.parse_args()

    path = f'Data/FullImages/{args.data}'
    images_paths = sorted(glob.glob(os.path.join(path, "image", "*.png")))
    # Extract names
    image_names = []
    for img in images_paths:
        image_names.append(img.split("\\")[-1].split(".")[0])
    print(f"{args.data}: {len(image_names)}")   
    for name in tqdm.tqdm(image_names, total=len(image_names)):
        img_path = f"Data/FullImages/{args.data}/image/{name}.png"
        img = cv2.imread(img_path)

        label_path = f"Data/FullImages/{args.data}/mask/{name}.png"
        label = cv2.imread(label_path)

        strided_crop(img, label, args.input_dim, args.input_dim,name,args.stride)