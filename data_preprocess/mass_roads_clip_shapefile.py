from osgeo import gdal
import os
import subprocess
from tqdm import tqdm


def clip_shapefile(raster_path, vector_path, save_path):
    # get the extent of the raster
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


def main():
    vector_path = "massachusetts_roads_shape.geojson"
    save_dir = 'train/shape'
    os.makedirs(save_dir, exist_ok=True)

    rasters_dir = 'train/sat/'
    rasters = os.listdir(rasters_dir)

    for raster in tqdm(rasters):
        raster_path = os.path.join(rasters_dir, raster)
        save_path = os.path.join(save_dir, raster.split('.')[0]+".geojson")
        clip_shapefile(raster_path, vector_path, save_path)


if __name__ == "__main__":
    main()
