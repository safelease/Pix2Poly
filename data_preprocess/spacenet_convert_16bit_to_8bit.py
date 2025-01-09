import os
from osgeo import gdal
from tqdm import tqdm


def convert_16bit_to_8bit(in_path, out_path, out_format='GTiff'):
    translate_options = gdal.TranslateOptions(format=out_format,
                                          outputType=gdal.GDT_Byte,
                                          scaleParams=[''],
                                          # scaleParams=[min_val, max_val],
                                          )
    gdal.Translate(destName=out_path, srcDS=in_path, options=translate_options)


def main():
    src_dir = f"../data/AOI_2_Vegas_Train/RGB-PanSharpen/"
    dest_dir = f"../data/AOI_2_Vegas_Train/RGB_8bit/train/images"
    os.makedirs(dest_dir, exist_ok=True)

    src_ims = os.listdir(src_dir)
    for im in tqdm(src_ims):
        src_im = os.path.join(src_dir, im)
        dest_im = os.path.join(dest_dir, im)
        convert_16bit_to_8bit(src_im, dest_im)


if __name__ == "__main__":
    main()

