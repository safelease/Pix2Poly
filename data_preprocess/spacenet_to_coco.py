# Modified from https://github.com/SarahwXU/HiSup/blob/main/tools/inria_to_coco.py
# Transform Spacenet 2 dataset (image and geojson pairs) to COCO format
#
# The first 15% images are kept as validation set

from pycocotools.coco import COCO
import os
import numpy as np
from skimage import io
import json
from tqdm import tqdm
from itertools import groupby
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as poly_transform
from shapely.ops import unary_union
from shapely.geometry import box
from skimage.measure import label as ski_label
from skimage.measure import regionprops
import cv2
import glob
import math
import shapely


def polygon2hbb(poly):
    """
    Get horizontal bounding box (match COCO)
    """
    p_x = poly[:, 0]
    p_y = poly[:, 1]
    hbb_x = np.min(p_x)
    hbb_y = np.min(p_y)
    hbb_w = np.around(np.max(p_x) - hbb_x, decimals=2)
    hbb_h = np.around(np.max(p_y) - hbb_y, decimals=2)
    hbox = [hbb_x, hbb_y, hbb_w, hbb_h]
    return [float(i) for i in hbox]

def clip_by_bound(poly, im_h, im_w):
    """
    Bound poly coordinates by image shape
    """
    p_x = poly[:, 0]
    p_y = poly[:, 1]
    p_x = np.clip(p_x, 0.0, im_w-1)
    p_y = np.clip(p_y, 0.0, im_h-1)
    return np.concatenate((p_x[:, np.newaxis], p_y[:, np.newaxis]), axis=1)

def crop2patch(im_p, p_h, p_w, p_overlap):
    """
    Get coordinates of upper-left point for image patch
    return: patch_list [X_upper-left, Y_upper-left, patch_width, patch_height]
    """
    im_h, im_w, _ = im_p
    x = np.arange(0, im_w-p_w, p_w-p_overlap)
    x = np.append(x, im_w-p_w)
    y = np.arange(0, im_h-p_h, p_h-p_overlap)
    y = np.append(y, im_h-p_h)
    X, Y = np.meshgrid(x, y)
    patch_list = [[i, j, p_w, p_h] for i, j in zip(X.flatten(), Y.flatten())]
    return patch_list

def polygon_in_bounding_box(polygon, bounding_box):
    """
    Returns True if all vertices of polygons are inside bounding_box
    :param polygon: [N, 2]
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    result = np.all(
        np.logical_and(
            np.logical_and(bounding_box[0] <= polygon[:, 0], polygon[:, 0] <= bounding_box[0] + bounding_box[2]),
            np.logical_and(bounding_box[1] <= polygon[:, 1], polygon[:, 1] <= bounding_box[1] + bounding_box[3])
        )
    )
    return result

def transform_poly_to_bounding_box(polygon, bounding_box):
    """
    Transform the original coordinates of polygon to bbox
    :param polygon: [N, 2]
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    transformed_polygon = polygon.copy()
    transformed_polygon[:, 0] -= bounding_box[0]
    transformed_polygon[:, 1] -= bounding_box[1]
    return transformed_polygon

def bmask_to_poly(b_im, simplify_ind, tolerance=1.8, ):
    """
    Convert binary mask to polygons
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    try:
        label_img = ski_label(b_im > 0)
    except:
        pass
        # print('error')
    props = regionprops(label_img)
    for prop in props:
        prop_mask = np.zeros_like(b_im)
        prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
        padded_binary_mask = np.pad(prop_mask, pad_width=1, mode='constant', constant_values=0)
        contours, hierarchy = cv2.findContours(padded_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            intp = []
            for contour, h in zip(contours, hierarchy[0]):
                contour = np.array([c.reshape(-1).tolist() for c in contour])
                # subtract pad
                contour -= 1
                contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
                if len(contour) > 3:
                    closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
                    if h[3] < 0:
                        extp = [tuple(i) for i in closed_c]
                    else:
                        if cv2.contourArea(closed_c.astype(int)) > 10:
                            intp.append([tuple(i) for i in closed_c])
            poly = Polygon(extp, intp)
            if simplify_ind:
                poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
                if isinstance(poly, Polygon):
                    polygons.append(poly)
                else:
                    for idx in range(len(poly.geoms)):
                        polygons.append(poly.geoms[idx])
        elif len(contours) == 1:
            contour = np.array([c.reshape(-1).tolist() for c in contours[0]])
            contour -= 1
            contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
            if len(contour) > 3:
                closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
                poly = Polygon(closed_c)

            # simply polygon vertex
                if simplify_ind:
                    poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
                if isinstance(poly, Polygon):
                    polygons.append(poly)
                else:
                    for idx in range(len(poly.geoms)):
                        polygons.append(poly.geoms[idx])
            # print(np.array(poly.exterior.coords).ravel().tolist())
            # in case that after "simplify", one polygon turn to multiply polygons
            # (pixels in polygon) are not connected
    return polygons

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def rotate_crop(im, gt, crop_size, angle):
    h, w = im.shape[0:2]
    im_rotated = rotate_image(im, angle)
    gt_rotated = rotate_image(gt, angle)
    if largest_rotated_rect(w, h, math.radians(angle))[0] >= crop_size:
        im_cropped = crop_around_center(im_rotated, crop_size, crop_size)
        gt_cropped = crop_around_center(gt_rotated, crop_size, crop_size)
    else:
        # print('error')
        im_cropped = crop_around_center(im, crop_size, crop_size)
        gt_cropped = crop_around_center(gt, crop_size, crop_size)
    return im_cropped, gt_cropped

def lt_crop(im, gt, crop_size):
    im_cropped = im[0:crop_size, 0:crop_size, :]
    gt_cropped = gt[0:crop_size, 0:crop_size]
    return im_cropped, gt_cropped

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

# for polygon vflip
def reflection():
    return lambda x, y: (x, -y)


if __name__ == '__main__':
    input_image_path = './Datasets/SpaceNet_v2_Building_detection/SN2_dataset/RGB_8bit/train/Vegas'
    input_annos_path = './Datasets/SpaceNet_v2_Building_detection/SN2_dataset/annotations/Vegas/pixel_geojson'

    save_path = './SpaceNet_v2_Building_detection/SN2_dataset/coco_format/Vegas'

    all_images = os.listdir(input_image_path)
    val_count = int(0.15 * len(all_images))
    print(f"No. of val images: {val_count}")
    val_images = all_images[0:val_count]
    train_images = all_images[val_count:]

    train_set = set(train_images)
    val_set = set(val_images)
    if len(train_set.intersection(val_set)) > 0 or len(val_set.intersection(train_set)):
        raise RuntimeError()

    split = 'train'

    if split == 'train':
        query_images = train_images
    elif split == 'val':
        query_images = val_images
    else:
        raise Exception(f'"{split}" is an incorrect split choice. Split choice must be either "train" or "val".')

    output_im_train = os.path.join(save_path, split, 'images')
    if not os.path.exists(output_im_train):
        os.makedirs(output_im_train)

    # patch_width = 725
    # patch_height = 725
    # patch_overlap = 300
    # patch_size = 512
    # rotation_list = [22.5, 45, 67.5]

    patch_width = 224
    patch_height = 224
    patch_overlap = 34  # ~15% of patch size
    patch_size = 224
    rotation_list = []

    # main dict for annotation file
    output_data_train = {
        'info': {'district': 'SpaceNetv2', 'description': 'building footprints', 'contributor': 'cyens'},
        'categories': [{'id': 100, 'name': 'building'}],
        'images': [],
        'annotations': [],
    }

    train_ob_id = 0
    train_im_id = 0
    # read in data with npy format
    input_label = os.listdir(input_annos_path)
    for g_id, label in enumerate(tqdm(input_label)):
        # read data
        # label_info = [''.join(list(g)) for k, g in groupby(label, key=lambda x: x.isdigit())]
        label_info = label.split('_')
        
        label_name = label_info[-1].split('.')[0]
        im_name = [im for im in all_images if label_name+".tif" in im]
        assert len(im_name) == 1
        im_name = im_name[0]
        image_data = io.imread(os.path.join(input_image_path, im_name))
        with open(os.path.join(input_annos_path, label), 'r') as f:
            anno_data = json.load(f)
        im_h, im_w, _ = image_data.shape

        tile_polygons = []
        for poly in anno_data['features']:
            poly = poly['geometry']['coordinates']
            assert len(poly) == 1
            poly = np.array(poly[0])
            poly = Polygon(poly)
            poly = poly_transform(reflection(), poly)
            tile_polygons.append(poly)
        tile_polygons = shapely.geometry.MultiPolygon(tile_polygons)
        tile_polygons = unary_union(tile_polygons)
        # tile_polygons = poly_transform(reflection(), tile_polygons)

        if im_name in query_images:
            # for training/val set, split image to 224x224
            patch_list = crop2patch(image_data.shape, patch_width, patch_height, patch_overlap)
            for pid, pa in enumerate(patch_list):
                x_ul, y_ul, pw, ph = pa
                # bbox_s = box(y_ul, y_ul+patch_height, x_ul, x_ul+patch_width)
                bbox_s = box(x_ul, y_ul, x_ul+patch_width, y_ul+patch_height)

                p_gt = tile_polygons.intersection(bbox_s)
                # print(type(p_gt))
                if isinstance(p_gt, Polygon):
                    p_gt = shapely.geometry.MultiPolygon([p_gt])
                else:
                    p_gt = shapely.geometry.MultiPolygon(p_gt)
                p_im = image_data[y_ul:y_ul+patch_height, x_ul:x_ul+patch_width, :]
                p_gts = []
                p_ims = []
                p_im_rd, _ = lt_crop(p_im, p_im[0], patch_size)
                p_gts.append(p_gt)
                p_ims.append(p_im_rd)
                # for angle in rotation_list:
                #     rot_im, _ = rotate_crop(p_im, p_im, patch_size, angle)
                #     # p_gts.append(rot_gt)
                #     p_ims.append(rot_im)
                for p_im, p_gt in zip(p_ims, p_gts):
                    if len(p_gt.geoms) > 0:
                        p_polygons = p_gt.geoms
                        for poly in p_polygons:
                            # poly = poly['geometry']['coordinates']
                            # assert len(poly) == 1
                            poly = np.asarray(poly.exterior.coords)
                            poly -= np.array([x_ul, y_ul])
                            poly = Polygon(poly)
                            # poly = poly_transform(reflection(), poly)
                            p_area = round(poly.area, 2)
                            if p_area > 0:
                                p_bbox = [poly.bounds[0], poly.bounds[1], poly.bounds[2]-poly.bounds[0], poly.bounds[3]-poly.bounds[1]]
                                if p_bbox[2] > 0 and p_bbox[3] > 0:
                                    p_seg = []
                                    coor_list = mapping(poly)['coordinates']
                                    assert len(coor_list) == 1
                                    # import code; code.interact(local=locals())
                                    for part_poly in coor_list:
                                        p_seg.append(np.asarray(part_poly).ravel().tolist())
                                    anno_info = {
                                        'id': train_ob_id,
                                        'image_id': train_im_id,
                                        'segmentation': p_seg,
                                        'area': p_area,
                                        'bbox': p_bbox,
                                        'category_id': 100,
                                        'iscrowd': 0
                                    }
                                    output_data_train['annotations'].append(anno_info)
                                    train_ob_id += 1
                    else:  # for including negative samples.
                        anno_info = {
                            'id': train_ob_id,
                            'image_id': train_im_id,
                            'segmentation': [],
                            'area': 0.,
                            'bbox': [],
                            'category_id': 100,
                            'iscrowd': 1
                        }
                        output_data_train['annotations'].append(anno_info)
                        train_ob_id += 1
                    # get patch info
                    p_name = label_name + '-' + str(train_im_id) + '.tif'
                    patch_info = {'id': train_im_id, 'file_name': p_name, 'width': patch_size, 'height': patch_size}
                    output_data_train['images'].append(patch_info)
                    # save patch image
                    io.imsave(os.path.join(output_im_train, p_name), p_im)
                    train_im_id += 1

    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(save_path)
    with open(os.path.join(save_path, split, 'annotation.json'), 'w') as f_json:
        json.dump(output_data_train, f_json)
