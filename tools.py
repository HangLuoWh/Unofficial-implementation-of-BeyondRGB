import torch
import numpy as np
import random
import os
import cv2
import math
from torchvision.utils import make_grid, save_image
import h5py
import json
import torch.nn.functional as F

# Very small value
EPS = 1e-8
# MS channel combinations given in the paper
TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6),
    (5, 6, 7), (6, 7, 8), (7, 8, 9), (8, 9, 10), (9, 10, 11),
    (10, 11, 12), (11, 12, 13), (12, 13, 14), (13, 14, 15),
    (0, 5, 9), (1, 6, 10), (2, 7, 11), (3, 8, 12), (4, 9, 13),
    (5, 10, 14), (3, 6, 15)
]

# Set the global random seed
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # Avoid selecting convolution algorithms, which reduces computational performance
    torch.backends.cudnn.deterministic = True  # Ensure the chosen convolution algorithm is non-random

# Create a directory
def make_dir(path):
    os.makedirs(path, exist_ok=True)

# Save an array as a PNG image
def show_array(img, path):
    img = img / img.max()
    img_8bit = (img*255).astype(np.uint8)
    cv2.imwrite(path, img_8bit)

def getMAE_numpy_2D(pred, gt):
    pred = pred / (np.linalg.norm(pred, axis=-1, ord=2, keepdims=True) + EPS)
    gt = gt / (np.linalg.norm(gt, axis=-1, ord=2, keepdims=True) + EPS)
    dot_product = (pred * gt).sum(axis=-1)
    clipped_result = np.clip(dot_product, -1 + EPS, 1 - EPS)
    radian = np.arccos(clipped_result)
    angle = (radian / np.pi) * 180
    return angle

# Demosaic multispectral image
def ms_demosaic(ms_raw):
    CFA_MAP = [
            [9, 11, 13, 15],
            [8, 10, 12, 14],
            [1, 3, 5, 7],
            [0, 2, 4, 6]
        ]
    h_phy, w_phy = ms_raw.shape
    h_eff = h_phy // 4
    w_eff = w_phy // 4
    ms_16ch = np.empty((h_eff, w_eff, 16), dtype=np.float32)  # Store the demosaiced image
    # Perform demosaicing by slicing
    for hw_row in range(4):
        for hw_col in range(4):
            ch = CFA_MAP[hw_row][hw_col]
            ms_16ch[:, :, ch] = ms_raw[hw_row::4, hw_col::4]

    return ms_16ch

# Process multispectral images for the training set
def process_ms_image(ms_h5_wt_path, ms_json, crop_size, crop_num, n_bins, hist_range):

    with h5py.File(ms_h5_wt_path, "r", driver='core') as f:
        ms_wt_raw = f['MIS'][:]

    # Demosaic the MS image
    ms_16ch_wt = ms_demosaic(ms_wt_raw)  # With labels

    # Extract the MS white point
    ms_wp = extract_ms_white_point(ms_16ch_wt, ms_json)

    # Remove the color chart from the image
    ms_wo_cc = remove_cc(ms_16ch_wt, ms_json)

    # Resize the image by its short side and randomly crop it
    ms_resize = resize_short_edge(ms_wo_cc, crop_size)
    ms_crops = random_crops(ms_resize, crop_size, crop_num)

    # Compute triplets for each image patch and generate the corresponding log-chroma histograms
    ms_crop_hist = generate_21_histograms(ms_crops, n_bins, hist_range)

    # Compute the UV cost map
    uv_cost_map = generate_uv_cost_maps(ms_wp, n_bins, hist_range)

    return ms_crop_hist, uv_cost_map

# Process multispectral images for validation/testing
def process_ms_image_2(ms_h5_wt_path, ms_json, crop_size, n_bins, hist_range):

    with h5py.File(ms_h5_wt_path, "r", driver='core') as f:
        ms_wt_raw = f['MIS'][:]

    # Demosaic the MS image
    ms_16ch_wt = ms_demosaic(ms_wt_raw)  # With labels

    # Extract the MS white point
    ms_wp = extract_ms_white_point(ms_16ch_wt, ms_json)

    # Remove the color chart from the image
    ms_wo_cc = remove_cc(ms_16ch_wt, ms_json)

    # Resize the image by its short side and then center-crop it
    ms_resize = resize_short_edge(ms_wo_cc, crop_size)
    ms_center = center_crop(ms_resize, crop_size)  # Center crop

    # Compute triplets for each image patch and generate the corresponding log-chroma histograms
    ms_crop_hist = generate_21_histograms(ms_center, n_bins, hist_range)

    # Compute the UV cost map
    uv_cost_map = generate_uv_cost_maps(ms_wp, n_bins, hist_range)

    return ms_crop_hist, uv_cost_map

# Random crop function
def random_crops(ms_np, crop_size, crop_number):
    # Determine whether the output size is an int or a list
    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size  # If it is an int, the crop height and width are the same
    else:
        crop_h, crop_w = crop_size

    H, W, _ = ms_np.shape  # Original image size

    crops = []
    
    for _ in range(crop_number):
        h_start = np.random.randint(0, H - crop_size + 1) 
        w_start = np.random.randint(0, W - crop_size + 1)
        crops.append(ms_np[h_start:h_start + crop_h, w_start:w_start + crop_w, :])

    return crops

# Resize by the short edge
def resize_short_edge(img, target_size):
    """
    Resize an array-type image by its short edge while preserving the aspect ratio.

    Args:
        img: np.ndarray, shape (H, W, C) or (H, W)
        target_size: int or tuple
            - int: target size is (target_size, target_size)

    Returns:
        Resized image
    """
    target_h = target_w = target_size

    h, w = img.shape[:2]

    # Compute the scaling factor so the short edge is resized to the target size
    scale = max(target_h / h, target_w / w)

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Resize each channel individually
    resized_channels = []
    for i in range(img.shape[2]):
        resized_channels.append(
            cv2.resize(img[..., i], (new_w, new_h), interpolation=cv2.INTER_AREA)
        )
    resized = np.stack(resized_channels, axis=-1)

    return resized

# Center crop
def center_crop(img, target_size):
    """
    Perform center cropping on an image.

    Args:
        img: np.ndarray
        target_size: int or tuple
            - int: crop to (target_size, target_size)
            - tuple: (height, width)

    Returns:
        Cropped image
    """
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size

    h, w = img.shape[:2]

    start_h = max(0, (h - target_h) // 2)
    start_w = max(0, (w - target_w) // 2)

    end_h = start_h + target_h
    end_w = start_w + target_w

    return [img[start_h:end_h, start_w:end_w, :]]

# Position of the MS white point
def get_valid_bbox(corners, img_h, img_w):
    xs = [int(c[0]) for c in corners]
    ys = [int(c[1]) for c in corners]
    # Find the upper-left and lower-right corner positions
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    # Ensure the coordinates remain within a reasonable range
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w - 1))
    y2 = max(y1 + 1, min(y2, img_h - 1))
    return x1, y1, x2, y2

# Extract the white point from the image
def extract_ms_white_point(ms_np, gt_json_path, target_patch="patch_20"):
    h, w = ms_np.shape[:2]
    with open(gt_json_path, "r", encoding="utf-8") as f:
        corner_20 = json.load(f)[target_patch]["corners"]

    corners_20_xy = [[round(x/4), round(y/4)] for x, y in corner_20]  # Divide coordinates by 4
    x1, y1, x2, y2 = get_valid_bbox(corners_20_xy, h, w)  # Obtain a feasible bounding box

    return np.median(ms_np[y1:y2, x1:x2], axis=(0, 1)).astype(np.float32)

# Remove the color chart from the multispectral image
def remove_cc(ms_np, gt_json_path):
    h, w = ms_np.shape[:2]
    all_corners = []  # Store all coordinates
    with open(gt_json_path, "r", encoding="utf-8") as f:
        json_f = json.load(f)
        # Get all coordinate positions to prepare for color-chart region detection
        for i in range(1, 25):
            for corordinate in json_f['patch_' + str(i)]['corners']:
                all_corners.append(corordinate)

    corners_all_xy = [[round(x/4), round(y/4)] for x, y in all_corners]  # Divide coordinates by 4
    x1_all, y1_all, x2_all, y2_all = get_valid_bbox(corners_all_xy, h, w)  # Color-chart coordinates

    ms_np[y1_all:y2_all, x1_all:x2_all, :] = 0
    return ms_np

# Read the real SPD values
def read_spd_from_cri(cri_txt_path):
    with open(cri_txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    spd_line = ""
    for idx, line in enumerate(lines):
        if "Illumination condition:" in line and "Emissive" in line:
            spd_line = lines[idx + 1]
            break

    str_vals = spd_line.strip(',').split(',')
    spd_values = np.array([float(v.strip()) for v in str_vals], dtype=np.float32)
    return spd_values

# Compute the log-chroma histogram
def compute_log_chroma_histogram(triplet, nbins, hist_range):
    c1, c2, c3 = triplet[:, :, 0], triplet[:, :, 1], triplet[:, :, 2]

    # Standard formula from the paper: u = log(c2/c1), v = log(c2/c3)
    c1 = np.maximum(c1, EPS)  # Prevent division by zero
    c3 = np.maximum(c3, EPS)  # Prevent division by zero
    u = np.log((c2) / (c1)).flatten()  # [H*W]
    v = np.log((c2) / (c3)).flatten()  # [H*W]
    luminance = np.linalg.norm(triplet, axis=2).flatten()  # Luminance

    min_val, max_val = hist_range
    bin_width = (max_val - min_val) / nbins

    # Build the 2D histogram
    hist, _, _ = np.histogram2d(u, v, bins=nbins,
                        range=((min_val - bin_width / 2, max_val + bin_width / 2),) * 2,
                        weights=luminance)

    # Root normalization to match the original CCC implementation
    norm_factor = hist.sum() + EPS
    hist = np.sqrt(hist / norm_factor)
    return hist

# Generate triplets and the corresponding log-chroma histograms for each image patch
def generate_21_histograms(ms_crops, nbins=128, hist_range=[-3.2, 3.2]):
    all_hist = []  # All histograms
    for i in range(len(ms_crops)):
        crop_hist = []
        for (c1, c2, c3) in TRIPLETS:
            triplet = ms_crops[i][:, :, [c1, c2, c3]]
            hist = compute_log_chroma_histogram(triplet, nbins, hist_range)
            crop_hist.append(hist)
        all_hist.append(crop_hist)
    return np.array(all_hist, dtype=np.float32)

# Generate the UV cost map
def generate_uv_cost_maps(ms_wp, n_bins, hist_range):
    ms_wp_triplet = get_21_uv_gt(ms_wp)  # Compute the triplets and corresponding UV chromaticity values for the MS white point
    H, _ = ms_wp_triplet.shape  # Number of triplets
    # Generate the grid
    u_cor = np.linspace(hist_range[0], hist_range[1], n_bins)
    v_cor = np.linspace(hist_range[0], hist_range[1], n_bins)
    # Generate grid coordinates and record the center point of each bin in the chroma histogram
    u_map, v_map = np.meshgrid(u_cor, v_cor)
    # Corresponds to paper formula 5
    u_map = np.exp(-np.expand_dims(u_map, axis=2))
    v_map = np.exp(-np.expand_dims(v_map, axis=2))
    one_map = np.ones_like(u_map)  # All-ones matrix
    exp_uv_map = np.concatenate((u_map, one_map, v_map), axis=2)

    # Compute all cost maps
    all_cost_map = np.zeros((H, n_bins, n_bins), dtype=np.float32)

    # Compute the cost map
    for i in range(H):
        gt_uv = ms_wp_triplet[i, :]  # Ground-truth UV values
        exp_gt_uv = np.exp(-gt_uv)
        exp_gt_uv = np.insert(exp_gt_uv, 1, 1)  # Insert 1
        exp_gt_uv = np.expand_dims(exp_gt_uv, (0, 1))  # Expand dimensions for broadcasting

        # Compute angular differences
        # Normalize
        exp_uv_map_norm = np.linalg.norm(exp_uv_map, axis=-1, keepdims=True) + EPS
        exp_gt_uv_norm = np.linalg.norm(exp_gt_uv, axis=-1, keepdims=True) + EPS

        exp_uv_map_unit = exp_uv_map / exp_uv_map_norm
        exp_gt_uv_unit = exp_gt_uv / exp_gt_uv_norm

        # Dot product point by point
        dot = np.sum(exp_uv_map_unit * exp_gt_uv_unit, axis=-1)

        # Restrict to [-1, 1] to avoid out-of-range arccos values
        dot = np.clip(dot, -1.0, 1.0)

        # Compute the angle (radians)
        angle_rad = np.arccos(dot)

        # Convert to degrees
        all_cost_map[i, :, :] = angle_rad * 180.0 / np.pi

    return all_cost_map

# Compute the ground-truth UV values for 21 triplets
def get_21_uv_gt(gt_ms):
    gt_uv = []
    for (c1, c2, c3) in TRIPLETS:
        g1 = gt_ms[c1] + EPS
        g2 = gt_ms[c2]
        g3 = gt_ms[c3] + EPS

        u = np.log(g2 / g1)
        v = np.log(g2 / g3)
        gt_uv.append(np.array([u, v]))
    return np.array(gt_uv)

if __name__ == '__main__':
    img_wt_path = 'beyondRGB/clb/BLUE_blue/1_light1/WT/MIS.h5'
    json_path = 'beyondRGB/clb/BLUE_blue/1_light1/WT/MIS_cc_detection.json'
    ms_crop_hist, ms_cost_map = process_ms_image(img_wt_path, json_path, 256, 4, 128, [-3.2, 3.2])

    show_array(ms_crop_hist[0, 0, :, :], 'image_crop_hist.png')
    show_array(ms_cost_map[0, :, :], 'image_cost_map.png')
    # show_array_as_gray(ms_cc[:,:,:3], 'image_cc.png')
    # show_array_as_gray(ms_wp[:,:,:3], 'image_wp.png')