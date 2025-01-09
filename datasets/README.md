# Datasets preparation

## INRIA Dataset

1. Download the [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/).
2. Extract and place the aerial image tiles in the `data` directory as follows:
```
data/inria_raw/
├── test/
│   └── images/
└── train/
    ├── gt/
    └── images/
```
3. Set path to raw INRIA train tiles and gts in L255 & L256 in `inria_to_coco.py`
4. Run the following command to prepare the INRIA dataset's train and validation splits in MS COCO format. The first 5 tiles of each city are kept as validation split as per the official recommendation.
```shell
# with pix2poly_env
python inria_to_coco.py
```
---


## SpaceNet 2 Building Detection v2 Dataset (Vegas Subset)

NOTE: We only use the Vegas subset for all our experiments in the paper.

1. Download the [Spacenet 2 Building Detection v2 Dataset](https://spacenet.ai/spacenet-buildings-dataset-v2/).
2. Extract and place the satellite image tiles for the Vegas subset in the `data` folder in the following directory structure:
```
data/AOI_2_Vegas_Train/
└── geojson/
    └── buildings/
└── RGB-PanSharpen/
    ├── gt/
    └── images/
```
3. Convert the pansharpened RGB image tiles from 16-bit to 8-bit using the following command:
```shell
# with gdal_env
python spacenet_convert_16bit_to_8bit.py
```
4. Convert geojson annotations from world space coordinates to pixel space coordinates using the following command:
```shell
# with gdal_env
python spacenet_world_to_pixel_coords.py
```
5. Set path to raw SpaceNet dataset's tiles and gts in L202 & L203 in `spacenet_to_coco.py`
6. Run the following command to prepare the SpaceNet dataset's train and validation splits in MS COCO format. The first 15% tiles kept as validation split.
```shell
# with pix2poly_env
python spacenet_to_coco.py
```
---


## WHU Buildings Dataset


1. Download the 0.2 meter split of the [WHU Buildings Aerial Imagery Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html).
2. Extract and place the aerial image tiles (512x512) in the `data` folder in the following directory structure:
```
data/WHU_aerial_0.2/
├── test/
│   ├── image/
│   └── label/
├── train/
│   ├── image/
│   └── label/
└── val/
    ├── image/
    └── label/
```
3. Set path to raw WHU Buildings tiles and gts (512x512) in L263 & L264 in `whu_buildings_to_coco.py`
4. Run the following command to prepare the WHU Buildings dataset's train, validation and test splits in MS COCO format.
```shell
# with pix2poly_env
python whu_buildings_to_coco.py
```
---



## Massachusetts Roads Dataset

1. Download the [Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/) using the following command:
```shell
./download_mass_roads_dataset.sh
```
2. Download and extract the roads vector shapefile for the dataset [here](https://www.cs.toronto.edu/~vmnih/data/mass_roads/massachusetts_roads_shape.zip). Use QGIS or any preferred tool to convert the SHP file to a geojson.
3. From this vector roads geojson, generate vector annotations in the image coordinate space for each image in the dataset by clipping the geojson to the corresponding raster extents:
```shell
python mass_roads_clip_shapefile.py
python mass_roads_world_to_pixel_coords.py
```
4. This results in the following directory structure containing the 1500x1500 tiles of the Massachusetts Roads Dataset:
```
mass_roads_1500/
├── test/
│   ├── map/
│   ├── pixel_geojson/
│   ├── sat/
│   └── shape/
├── train/
│   ├── map/
│   ├── pixel_geojson/
│   ├── sat/
│   └── shape/
└── valid/
    ├── map/
    ├── pixel_geojson/
    ├── sat/
    └── shape/
```
5. Split the 1500x1500 tiles into 224x224 overlapping patches with the following command:
```shell
python mass_roads_tiles_to_patches.py
```
6. Generate vector annotation files for the patches as follows:
```shell
python mass_roads_clip_tile_vectors.py
```
7. This results in the processed 224x224 patches of the Massachusetts Roads Dataset to be used for training Pix2Poly in the following directory structure:
```
data/mass_roads_224/
├── test/
│   ├── map/
│   ├── pixel_geojson/
│   ├── sat/
│   └── shape/
├── train/
│   ├── map/
│   ├── pixel_geojson/
│   ├── sat/
│   └── shape/
└── valid/
    ├── map/
    ├── pixel_geojson/
    ├── sat/
    └── shape/
```
---
