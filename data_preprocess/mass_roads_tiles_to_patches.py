# gdal_retile.py -targetDir gdal_retile/valid/images/ -ps 224 224 -overlap 34 valid/sat/*.tiff

import os
import subprocess
from tqdm import tqdm


def main():
    split = 'valid'
    images_dir = f"data/mass_roads/{split}/map"

    out_dir = f"data/mass_roads/gdal_retile/{split}/mask"
    os.makedirs(out_dir, exist_ok=True)

    patch_size = 224
    ph = pw = patch_size
    overlap = int(round(0.15*patch_size))

    images = os.listdir(images_dir)
    for img in tqdm(images):
        in_path = os.path.join(images_dir, img)
        cmd = f"gdal_retile.py -targetDir {out_dir} -ps {ph} {pw} -overlap {overlap} {in_path}"
        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()
