import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MyLoss(nn.Module):
    def __init__(self, hyper):
        super(MyLoss, self).__init__()
        self.hyper = hyper
        self.w1 = self.hyper['TRAIN']['LOSS']['W1']
        self.w2 = self.hyper['TRAIN']['LOSS']['W2']

    def angular_error(self, pred, gt):
        eps = 1e-6

        # Ensure the input dimensions match
        assert pred.shape == gt.shape, "Pred and GT must have the same shape"

        # Compute the dot product and vector lengths
        dot_product = torch.sum(pred * gt, dim=1)
        pred_length = torch.norm(pred, p=2, dim=1) + eps  # Avoid division by zero
        gt_length = torch.norm(gt, p=2, dim=1) + eps

        # Compute cosine similarity and apply safe clipping
        cosine_sim = dot_product / (pred_length * gt_length)

        # Use a safer clipping range to keep values within [-1, 1]
        # Floating-point precision may require a slightly narrower range
        cosine_sim = torch.clamp(cosine_sim, -1 + eps, 1 - eps)

        # Compute the angle (radians)
        angle_rad = torch.acos(cosine_sim)

        # Convert to degrees and compute the mean
        angle_deg = angle_rad * (180.0 / math.pi)
        return angle_deg

    def forward(self, f_score_maps, cost_maps, pred_spd, gt_spd):
        Lccc_i = torch.sum(f_score_maps * cost_maps, dim=(2, 3))  # Corresponds to formula 7
        Lccc = torch.sum(Lccc_i, dim=1)
        Lcnn = self.angular_error(pred_spd, gt_spd)  # Compute the angular difference
        total_loss = self.w1 * Lccc + self.w2 * Lcnn  # Compute the overall error
        return torch.sum(total_loss).item(), torch.mean(total_loss)  # Sum of losses across all samples and the mean loss