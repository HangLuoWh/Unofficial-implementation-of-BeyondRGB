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

# 极小值
EPS = 1e-8
#论文中给出的ms通道组合
TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6),
    (5, 6, 7), (6, 7, 8), (7, 8, 9), (8, 9, 10), (9, 10, 11),
    (10, 11, 12), (11, 12, 13), (12, 13, 14), (13, 14, 15),
    (0, 5, 9), (1, 6, 10), (2, 7, 11), (3, 8, 12), (4, 9, 13),
    (5, 10, 14), (3, 6, 15)
]

# 设定整体程序的随机数种子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False # 避免选择卷积算法，会降低运算性能
    torch.backends.cudnn.deterministic = True # 保证所选用的卷积算法是非随机的

# 创建文件夹
def make_dir(path):
    os.makedirs(path,exist_ok=True)

# 将一个array保存为一张png图像
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

# 多光谱图像去马赛克
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
    ms_16ch = np.empty((h_eff, w_eff, 16), dtype=np.float32) #  存放区马赛克后的图像
    # 通过切片完成去马赛克
    for hw_row in range(4):
        for hw_col in range(4):
            ch = CFA_MAP[hw_row][hw_col]
            ms_16ch[:, :, ch] = ms_raw[hw_row::4, hw_col::4]

    return ms_16ch

# 处理训练集上的多光谱图像
def process_ms_image(ms_h5_wt_path, ms_json, crop_size, crop_num, n_bins, hist_range):

    with h5py.File(ms_h5_wt_path, "r", driver='core') as f:
        ms_wt_raw = f['MIS'][:]

    # ms去马赛克
    ms_16ch_wt = ms_demosaic(ms_wt_raw) # 带标注

    # 获取ms的白点
    ms_wp = extract_ms_white_point(ms_16ch_wt, ms_json)

    # 删除图像上色卡
    ms_wo_cc = remove_cc(ms_16ch_wt, ms_json)

    # # 对图像作短边等比缩放及随机裁切
    ms_resize = resize_short_edge(ms_wo_cc, crop_size)
    ms_crops = random_crops(ms_resize, crop_size, crop_num)

    # 计算图像块的三元组并生成对应的对数色度直方图
    ms_crop_hist = generate_21_histograms(ms_crops, n_bins, hist_range)

    # 计算uv cost map
    uv_cost_map = generate_uv_cost_maps(ms_wp, n_bins, hist_range)

    return ms_crop_hist, uv_cost_map

# 处理验证/测试集上的多光谱图像
def process_ms_image_2(ms_h5_wt_path, ms_json, crop_size, n_bins, hist_range):

    with h5py.File(ms_h5_wt_path, "r", driver='core') as f:
        ms_wt_raw = f['MIS'][:]

    # ms去马赛克
    ms_16ch_wt = ms_demosaic(ms_wt_raw) # 带标注

    # 获取ms的白点
    ms_wp = extract_ms_white_point(ms_16ch_wt, ms_json)

    # 删除图像上色卡
    ms_wo_cc = remove_cc(ms_16ch_wt, ms_json)

    # 对图像作短边等比缩放，然后作中心裁切
    ms_resize = resize_short_edge(ms_wo_cc, crop_size)
    ms_center = center_crop(ms_resize, crop_size) # 作中心裁切

    # 计算图像块的三元组并生成对应的对数色度直方图
    ms_crop_hist = generate_21_histograms(ms_center, n_bins, hist_range)

    # 计算uv cost map
    uv_cost_map = generate_uv_cost_maps(ms_wp, n_bins, hist_range)

    return ms_crop_hist, uv_cost_map

# 随机裁切函数
def random_crops(ms_np, crop_size, crop_number):
    # 判断输出的尺寸是一个数字还是一个列表
    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size # 若是数字则裁切的长宽相同
    else:
        crop_h, crop_w = crop_size

    H, W, _ = ms_np.shape # 原图的尺寸

    crops = []
    
    for _ in range(crop_number):
        h_start = np.random.randint(0, H - crop_size + 1) 
        w_start = np.random.randint(0, W - crop_size + 1)
        crops.append(ms_np[h_start:h_start + crop_h, w_start:w_start + crop_w, :])

    return crops

# 短边等比缩放
def resize_short_edge(img, target_size):
    """
    对 array 类型图像做短边等比缩放。
    
    参数:
        img: np.ndarray, 形状为 (H, W, C) 或 (H, W)
        target_size: int 或 tuple
            - int: 目标尺寸为 (target_size, target_size)
    
    返回:
        缩放后的图像
    """
    target_h = target_w = target_size

    h, w = img.shape[:2]

    # 计算缩放比例，使短边缩放到目标尺寸
    scale = max(target_h / h, target_w / w)

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # 逐通道作缩放
    resized_channels = []
    for i in range(img.shape[2]):
        resized_channels.append(
            cv2.resize(img[..., i], (new_w, new_h), interpolation=cv2.INTER_AREA)
        )
    resized = np.stack(resized_channels, axis=-1)

    return resized

# 中心裁切
def center_crop(img, target_size):
    """
    对图像进行中心裁切。
    
    参数:
        img: np.ndarray
        target_size: int 或 tuple
            - int: 裁切为 (target_size, target_size)
            - tuple: (height, width)
    
    返回:
        裁切后的图像
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

# 多光谱白点的位置
def get_valid_bbox(corners, img_h, img_w):
    xs = [int(c[0]) for c in corners]
    ys = [int(c[1]) for c in corners]
    # 找到左上方和右下方的坐标位置
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    # 确保坐标在合理的范围内
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w - 1))
    y2 = max(y1 + 1, min(y2, img_h - 1))
    return x1, y1, x2, y2

# 抽取图像中的白点
def extract_ms_white_point(ms_np, gt_json_path, target_patch="patch_20"):
    h, w = ms_np.shape[:2]
    with open(gt_json_path, "r", encoding="utf-8") as f:
        corner_20 = json.load(f)[target_patch]["corners"]
    
    corners_20_xy = [[round(x/4), round(y/4)] for x, y in corner_20] # 将坐标除4
    x1, y1, x2, y2 = get_valid_bbox(corners_20_xy, h, w) # 获取可行的方框位置

    return np.median(ms_np[y1:y2, x1:x2], axis=(0, 1)).astype(np.float32)

# 消除多光谱图像上的色卡
def remove_cc(ms_np, gt_json_path):
    h, w = ms_np.shape[:2]
    all_corners = [] # 存放所有的坐标
    with open(gt_json_path, "r", encoding="utf-8") as f:
        json_f = json.load(f)
        # 获取所有的坐标位置，为找色卡区域做准备
        for i in range(1, 25):
            for corordinate in json_f['patch_' + str(i)]['corners']:
                all_corners.append(corordinate)

    corners_all_xy = [[round(x/4), round(y/4)] for x, y in all_corners] # 将坐标除4
    x1_all, y1_all, x2_all, y2_all = get_valid_bbox(corners_all_xy, h, w) # 色卡的坐标

    ms_np[y1_all:y2_all, x1_all:x2_all, :] = 0
    return ms_np

# 获取真实的SPD值
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

# 计算对数色度直方图
def compute_log_chroma_histogram(triplet, nbins, hist_range):
    c1, c2, c3 = triplet[:, :, 0], triplet[:, :, 1], triplet[:, :, 2]

    # 论文标准公式：u = log(c2/c1), v = log(c2/c3)
    c1 = np.maximum(c1, EPS) # 防止0除
    c3 = np.maximum(c3, EPS) # 防止0除
    u = np.log((c2) / (c1)).flatten() # [H*W]
    v = np.log((c2) / (c3)).flatten()  # [H*W]
    luminance = np.linalg.norm(triplet, axis=2).flatten()  # 亮度

    min_val, max_val = hist_range # 
    bin_width = (max_val - min_val) / nbins # 

    # 统计二维直方图
    hist, _, _ = np.histogram2d(u, v, bins = nbins,\
                        range = ((min_val - bin_width / 2, max_val + bin_width / 2),) * 2, \
                        weights = luminance)
    
    # 平方根归一化，对齐CCC原版
    norm_factor = hist.sum() + EPS
    hist = np.sqrt(hist / norm_factor)
    return hist

# 为每一个图像块生成三元组并计算对应的对数色度直方图
def generate_21_histograms(ms_crops, nbins=128, hist_range = [-3.2, 3.2]):
    all_hist = [] # 所有的直方图
    for i in range(len(ms_crops)):
        crop_hist = []
        for (c1, c2, c3) in TRIPLETS:
            triplet = ms_crops[i][:, :, [c1, c2, c3]]
            hist = compute_log_chroma_histogram(triplet, nbins, hist_range)
            crop_hist.append(hist)
        all_hist.append(crop_hist)
    return np.array(all_hist, dtype=np.float32)

# 生成uv cost map
def generate_uv_cost_maps(ms_wp, n_bins, hist_range):
    ms_wp_triplet = get_21_uv_gt(ms_wp) # 计算多光谱白点的三元组以及对应的uv色度值
    H, _ = ms_wp_triplet.shape # 获得三元组的数量
    # 生成网格
    u_cor = np.linspace(hist_range[0], hist_range[1], n_bins)
    v_cor = np.linspace(hist_range[0], hist_range[1], n_bins)
    # 生成网格坐标点，记录色度直方图内每个bin的中心点坐标
    u_map, v_map = np.meshgrid(u_cor, v_cor)
    # 对应论文的公式5
    u_map = np.exp(-np.expand_dims(u_map, axis = 2))
    v_map = np.exp(-np.expand_dims(v_map, axis = 2))
    one_map = np.ones_like(u_map) # 全为1的矩阵
    exp_uv_map = np.concatenate((u_map, one_map, v_map), axis = 2)

    # 计算所有的cost map
    all_cost_map = np.zeros((H, n_bins, n_bins), dtype=np.float32)

    # 计算cost map
    for i in range(H):
        gt_uv = ms_wp_triplet[i, :] # 真实的uv值
        exp_gt_uv = np.exp(-gt_uv)
        exp_gt_uv = np.insert(exp_gt_uv, 1, 1) # 插入1
        exp_gt_uv = np.expand_dims(exp_gt_uv, (0, 1)) # 拓展维度便于广播

        # 计算角度差
        # 归一化
        exp_uv_map_norm = np.linalg.norm(exp_uv_map, axis=-1, keepdims = True) + EPS
        exp_gt_uv_norm = np.linalg.norm(exp_gt_uv, axis=-1, keepdims = True) + EPS

        exp_uv_map_unit = exp_uv_map / exp_uv_map_norm
        exp_gt_uv_unit = exp_gt_uv / exp_gt_uv_norm

        # 逐点点积
        dot = np.sum(exp_uv_map_unit * exp_gt_uv_unit, axis=-1)

        # 限制到 [-1, 1]，避免 arccos 取值越界
        dot = np.clip(dot, -1.0, 1.0)

        # 计算夹角（弧度）
        angle_rad = np.arccos(dot)

        # 转成角度
        all_cost_map[i, :, :] = angle_rad * 180.0 / np.pi

    return all_cost_map

# 计算21组三元组的真值uv
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