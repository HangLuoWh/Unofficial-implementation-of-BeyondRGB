import torch
from torch.utils.data import Dataset
import os
import tools
import re

class CustomDataset(Dataset):
    def __init__(self, hyper):
        self.hyper = hyper
        self.ms_size = hyper['TRAIN']['MS_SIZE']
        self.crop_num = hyper['TRAIN']['NUMBER']
        self.n_bins = hyper['TRAIN']['N_BINS']
        self.hist_range = hyper['TRAIN']['HIST_RANGE']
        self.train_txt = './BeyondRGB_Data/train_dataset_paper_git.txt'
        # 读取训练集列表
        with open(self.train_txt, 'r', encoding='utf-8') as f:
            self.train_list = [line.strip() for line in f if line.strip()]

        print(f"训练集总数：{len(self.train_list)}")

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        scene_path = self.train_list[idx]
        scene_path = re.sub(r"_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", scene_path) # 用正则表达式去掉中间的时间
        scene_path = os.path.join(self.hyper['TRAIN']['DATASET_PATH'], scene_path)

        ms_h5_path_wt = os.path.join(scene_path, "WT", "MIS.h5")
        ms_json = os.path.join(scene_path, "WT", "MIS_cc_detection.json")
        cri_path = os.path.join(scene_path, "CRI","CRI.txt")

        # 对多光谱图像作随机裁切-提取三元组-计算对数色度直方图-计算三元组的cost maps
        ms_crop_hist, ms_cost_map = tools.process_ms_image(ms_h5_path_wt, ms_json, \
                                                        self.ms_size, self.crop_num, \
                                                        self.n_bins, self.hist_range)

        # 读取SPD真值
        gt_spd_np = tools.read_spd_from_cri(cri_path)

        return {
            "ms_crop_hist": torch.from_numpy(ms_crop_hist),
            "ms_cost_map": torch.from_numpy(ms_cost_map),
            "gt_spd": torch.from_numpy(gt_spd_np)
        }

# ===================== 验证集 Dataset =====================
class CustomValDataset(Dataset):
    def __init__(self, hyper):
        self.hyper = hyper
        self.ms_size = hyper['TRAIN']['MS_SIZE']
        self.crop_num = hyper['TRAIN']['NUMBER']
        self.n_bins = hyper['TRAIN']['N_BINS']
        self.hist_range = hyper['TRAIN']['HIST_RANGE']
        self.val_txt = './BeyondRGB_Data/val_dataset_paper_git.txt'
        with open(self.val_txt, 'r', encoding='utf-8') as f:
            self.val_list = [line.strip() for line in f if line.strip()]

        print(f"验证集总数：{len(self.val_list)}")

    def __len__(self):
        return len(self.val_list)

    def __getitem__(self, idx):
        scene_path = self.val_list[idx]
        scene_path = re.sub(r"_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", scene_path) # 用正则表达式去掉中间的时间
        scene_path = os.path.join(self.hyper['TRAIN']['DATASET_PATH'], scene_path)

        ms_h5_path_wt = os.path.join(scene_path, "WT", "MIS.h5")
        ms_json = os.path.join(scene_path, "WT", "MIS_cc_detection.json")
        cri_path = os.path.join(scene_path, "CRI","CRI.txt")

        # 对多光谱图像作随机裁切-提取三元组-计算对数色度直方图-计算三元组的cost maps
        ms_crop_hist, ms_cost_map = tools.process_ms_image_2(ms_h5_path_wt, ms_json, \
                                                        self.ms_size, self.n_bins, \
                                                        self.hist_range)

        # 读取SPD真值
        gt_spd_np = tools.read_spd_from_cri(cri_path)

        return {
            "ms_crop_hist": torch.from_numpy(ms_crop_hist),
            "ms_cost_map": torch.from_numpy(ms_cost_map),
            "gt_spd": torch.from_numpy(gt_spd_np)
        }

if __name__ == '__main__':
    import yaml
    with open("hyper_parameters.yaml", "r", encoding="utf-8") as f:
        hyper = yaml.safe_load(f)
    ds = CustomDataset(hyper)
    ds1 = CustomValDataset(hyper)
    sample = ds[0]
    sample1 = ds1[0]
    print("ms_crop_hist  shape:", sample["ms_crop_hist"].shape)
    print("ms_cost_map value:", sample["ms_cost_map"].shape)
    print("gt_spd value:", sample["gt_spd"].shape)