import cv2 
import numpy as np 
import glob 
import os 
import argparse 
import tqdm 
 

def clahe_3d (img,gs): 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab) 
    clahe = cv2.createCLAHE(clipLimit=8,tileGridSize=(gs,gs)) 
    img[:,:,0] = clahe.apply(img[:,:,0]) 
    img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB) 
    return img 

  
if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--usecase', type=str, default='filter', required=False, choices=['filter' , 'process']) 
    parser.add_argument('--data', type=str, default='train', required=False, choices=['train' , 'valid', 'test']) 
    parser.add_argument('--CLAHE', type=str, required=False,  default='no', choices=['yes','no'])
    args = parser.parse_args()  
 
    if args.usecase == 'filter': 
        path = f'Data/Crops/{args.data}' 
        images_paths = sorted(glob.glob(os.path.join(path, "labels", "*.png"))) 
        i = 0 
        for label in tqdm.tqdm(images_paths, total=len(images_paths)): 
            name = label.split("\\")[-1].split(".")[0] 
            label_arr = cv2.imread(label, cv2.IMREAD_GRAYSCALE) 
            if len(np.unique(label_arr)) == 1: 
                os.remove(label) 
                os.remove(f'Data/Crops/{args.data}/image/{name}.png') 
                i+=1 
        print(f"{i} many empty patches from {len(images_paths)} deleted.") 
    else: 
        path = f'Data/FullImages/{args.data}' 
        images_paths = sorted(glob.glob(os.path.join(path, "image_org", "*.png"))) 
        if args.CLAHE == 'yes':
            for img in tqdm.tqdm(images_paths, total=len(images_paths)): 
                name = img.split("\\")[-1].split(".")[0] 
                img = cv2.imread(img, cv2.IMREAD_COLOR)  
                sclera = cv2.imread(f"Data/FullImages/{args.data}/mask_sclera/{name}.png", cv2.IMREAD_GRAYSCALE)
                
                img = clahe_3d(img,50) 
                sclera = np.expand_dims(sclera/255, axis=-1)  

                cv2.imwrite(f'Data/FullImages/{args.data}/image/{name}.png',img*sclera)
        else:
            for img in tqdm.tqdm(images_paths, total=len(images_paths)): 
                name = img.split("\\")[-1].split(".")[0] 
                img = cv2.imread(img, cv2.IMREAD_COLOR)  
                sclera = cv2.imread(f"Data/FullImages/{args.data}/mask_sclera/{name}.png", cv2.IMREAD_GRAYSCALE) /255  
                sclera = np.expand_dims(sclera, axis=-1)  
                cv2.imwrite(f'Data/FullImages/{args.data}/image/{name}.png',img*sclera)