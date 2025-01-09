from osgeo import gdal
import os
import subprocess
from tqdm import tqdm
from multiprocessing import Pool


def clip_shapefile(paths_dict):
    # get the extent of the raster
    raster_path = paths_dict['raster_path']
    save_path = paths_dict['save_path']
    vector_path = paths_dict['vector_path']
    src = gdal.Open(raster_path)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    sizeX = src.RasterXSize * xres
    sizeY = src.RasterYSize * yres
    lrx = ulx + sizeX
    lry = uly + sizeY
    src = None

    # format the extent coords
    extent = f"{ulx} {lry} {lrx} {uly}"
    # print(extent)

    # make clip command with ogr2ogr
    cmd = f"ogr2ogr {save_path} {vector_path} -clipsrc {extent}"

    # call the command
    subprocess.call(cmd, shell=True)
    return 0


def main():
    split = "train"  # "train" or "valid" or "test"
    data_root = "../data/mass_roads_1500"
    vector_path = os.path.join(data_root, "massachusetts_roads_shape.geojson")
    save_dir = os.path.join(data_root, split, "shape")
    os.makedirs(save_dir, exist_ok=True)

    rasters_dir = os.path.join(data_root, split, "sat")
    rasters = os.listdir(rasters_dir)

    param_inputs = []
    for raster in rasters:
        raster_path = os.path.join(rasters_dir, raster)
        save_path = os.path.join(save_dir, raster.split('.')[0]+".geojson")
        param_inputs.append(
            {
                'raster_path': raster_path,
                'vector_path': vector_path,
                'save_path': save_path,
            }
        )
        # clip_shapefile(raster_path, vector_path, save_path)

    with Pool() as p:
        _ = list(tqdm(p.imap(clip_shapefile, param_inputs), total=len(param_inputs)))


if __name__ == "__main__":
    main()

