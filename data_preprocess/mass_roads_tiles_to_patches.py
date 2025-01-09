# gdal_retile.py -targetDir gdal_retile/valid/images/ -ps 224 224 -overlap 34 valid/sat/*.tiff

import os
import subprocess
from tqdm import tqdm


def main():
    split = 'train'
    data_root = f"../data/mass_roads_1500"
    images_dir = os.path.join(data_root, split, "sat")
    masks_dir = os.path.join(data_root, split, "map")

    out_imgs_dir = f"../data/mass_roads_224/{split}/images"
    out_mask_dir = f"../data/mass_roads_224/{split}/mask"
    os.makedirs(out_imgs_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    patch_size = 224
    ph = pw = patch_size
    overlap = int(round(0.15*patch_size))

    images = os.listdir(images_dir)
    for img in tqdm(images):
        in_path = os.path.join(images_dir, img)
        cmd = f"gdal_retile.py -targetDir {out_imgs_dir} -ps {ph} {pw} -overlap {overlap} {in_path}"
        subprocess.call(cmd, shell=True)

    masks = os.listdir(masks_dir)
    for mask in tqdm(masks):
        in_path = os.path.join(masks_dir, mask)
        cmd = f"gdal_retile.py -targetDir {out_mask_dir} -ps {ph} {pw} -overlap {overlap} {in_path}"
        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()
