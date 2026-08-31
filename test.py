import os
import yaml
import numpy as np
import torch
import time
from tqdm import tqdm
import model
import tools
import re

class Tester:
    def __init__(self, log_path, config_file_name):
        self.log_path = log_path
        # Load the hyperparameters saved by training
        config_full_path = os.path.join(log_path, config_file_name)
        with open(config_full_path, 'r', encoding="utf-8") as f:
            self.hyper = yaml.safe_load(f)

        # Compute device
        self.device = torch.device(self.hyper['TRAIN']['DEVICE'])

        # Initialize the model and load weights
        self.model = getattr(model, self.hyper["MODEL"]["NAME"])()
        self.load()  # Load model weights
        self.model = self.model.to(self.device)
        self.model.eval()

    def load(self):
        ckpt_path = os.path.join(self.log_path, "model_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model weights do not exist: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=True)
        print(f"Successfully loaded the best model: {ckpt_path}")

    def test(self):
        with open('BeyondRGB_Data/val_dataset_paper_git.txt', "r", encoding="utf-8") as f:
            self.test_list = [line.strip() for line in f if line.strip()]

        # Split the Lab dataset and Field dataset
        self.lab_test_list = self.test_list[:121]
        self.field_test_list = self.test_list[121:]

        print("============== Start testing Lab dataset ==============")
        lab_records, lab_fps = self._test_single_subset(self.lab_test_list, "Lab")

        print("\n============== Start testing Field dataset ==============")
        field_records, field_fps = self._test_single_subset(self.field_test_list, "Field")

        print(f"\n============== Testing complete ==============")
        print(f"Lab valid sample count: {len(lab_records)}, FPS: {lab_fps:.2f}")
        print(f"Field valid sample count: {len(field_records)}, FPS: {field_fps:.2f}")

    def _test_single_subset(self, data_list, subset_name):
        records = []
        time_list = []

        for scene_path in tqdm(data_list, desc=f"Testing {subset_name}"):
            scene_path = re.sub(r"_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", scene_path)  # Use regex to remove the timestamp in the middle
            scene_path = os.path.join(self.hyper['TRAIN']['DATASET_PATH'], scene_path)

            ms_h5_path = os.path.join(scene_path, "WT", "MIS.h5")
            ms_json = os.path.join(scene_path, "WT", "MIS_cc_detection.json")
            cri_path = os.path.join(scene_path, "CRI", "CRI.txt")

            ms_crop_hist, _ = tools.process_ms_image_2(ms_h5_path, ms_json, \
                                                self.hyper['TRAIN']['MS_SIZE'],\
                                                self.hyper['TRAIN']['N_BINS'], \
                                                self.hyper['TRAIN']['HIST_RANGE'])
            # Read the ground-truth SPD
            gt_spd_np = tools.read_spd_from_cri(cri_path)

            # ===================== Model inference =====================
            t0 = time.time()
            with torch.no_grad():
                _, spd_pred = self.model(torch.from_numpy(ms_crop_hist).to(self.device))
            cost = time.time() - t0

            # Convert to NumPy and compute metrics
            pred_np = spd_pred.detach().cpu().numpy().squeeze(0)
            angle = tools.getMAE_numpy_2D(pred_np, gt_spd_np)
            records.append({'angle': angle})  # Record the angle
            time_list.append(cost)  # Record the time cost

        # Calculate FPS
        total_time = sum(time_list)
        fps = len(time_list) / total_time if total_time > 0 else 0.0
        print(f"\n[{subset_name} dataset] valid samples: {len(records)}, FPS: {fps:.2f}")

        if len(records) > 0:
            self.statistic(records, subset_name)
        else:
            print(f"{subset_name} has no valid test samples!")
        return records, fps

    def statistic(self, records, subset_name):
        records_sorted = sorted(records, key=lambda x: x["angle"])
        angle_list = [item["angle"] for item in records_sorted]
        n = len(angle_list)

        # Compute evaluation metrics
        mean_angle = np.mean(angle_list)
        std_angle = np.std(angle_list)
        median_angle = np.median(angle_list)
        q1, q3 = np.percentile(angle_list, [25, 75])
        trimean = (2 * median_angle + q1 + q3) / 4
        best_25 = np.mean(angle_list[:max(1, int(n * 0.25))])
        tail_idx = int(n * 0.75)
        worst_25 = np.mean(angle_list[tail_idx:]) if tail_idx < n else angle_list[-1]
        per95 = np.percentile(angle_list, 95)

        # Print metrics
        print(f"\n[{subset_name} dataset statistics]")
        print(f"Mean: {mean_angle:.4f}°")
        print(f"Std: {std_angle:.4f}°")
        print(f"Median: {median_angle:.4f}°")
        print(f"Trimean: {trimean:.4f}°")
        print(f"Best25: {best_25:.4f}°")
        print(f"Worst25: {worst_25:.4f} °")
        print(f"95%: {per95:.4f}°")

if __name__ == '__main__':
    # Update this to your training log directory path
    MODEL_PATH = "./Result/BeyondRGB_experiment/08301216"
    tester = Tester(log_path = MODEL_PATH, config_file_name="hyper_parameters.yaml")
    tester.test()