import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- BeyondRGB Network -----------------
class CCCPyramid(nn.Module):
    # Corresponds to paper Table S2
    def __init__(self):
        super(CCCPyramid, self).__init__()
        # 7 layers of 5x5 convolution, corresponding to Conv1~Conv7 in Table S2
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(1, 1, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(1, 1, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(1, 1, kernel_size=5, padding=2)

        # Gaussian blur kernel defined in paper formula (S1)
        self.register_buffer(
            'blur_kernel',
            torch.tensor([
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ], dtype=torch.float32).view(1, 1, 3, 3)
        )

    def _blur(self, x):
        # Corresponding Gaussian blur operation
        return F.conv2d(x, self.blur_kernel, padding=1)

    def forward(self, x: torch.Tensor):
        # ========== Downsampling path ==========
        c1 = self.conv1(x)
        x = F.interpolate(c1, scale_factor=0.5, mode='bilinear', align_corners=False)  # 128→64

        c2 = self.conv2(x)
        x = F.interpolate(c2, scale_factor=0.5, mode='bilinear', align_corners=False)  # 64→32

        c3 = self.conv3(x)
        x = F.interpolate(c3, scale_factor=0.5, mode='bilinear', align_corners=False)  # 32→16

        c4 = self.conv4(x)
        x = F.interpolate(c4, scale_factor=0.5, mode='bilinear', align_corners=False)  # 16→8

        c5 = self.conv5(x)
        x = F.interpolate(c5, scale_factor=0.5, mode='bilinear', align_corners=False)  # 8→4

        c6 = self.conv6(x)
        x = F.interpolate(c6, scale_factor=0.5, mode='bilinear', align_corners=False)  # 4→2

        c7 = self.conv7(x)

        # ========== 6 upsampling layers + skip connections ==========
        out = F.interpolate(c7, scale_factor=2, mode='bilinear', align_corners=False)  # 2→4
        out = self._blur(out)
        out = out + c6  # Skip connection

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)  # 4→8
        out = self._blur(out)
        out = out + c5

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)  # 8→16
        out = self._blur(out)
        out = out + c4

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)  # 16→32
        out = self._blur(out)
        out = out + c3

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)  # 32→64
        out = self._blur(out)
        out = out + c2

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)  # 64→128
        out = self._blur(out)
        out = out + c1

        # Global softmax normalization
        B, C, H, W = out.shape
        out = out.flatten(1).softmax(dim=1).view(B, C, H, W)
        return out

class ISEmodel(nn.Module):
    def __init__(self, num_triplets: int = 21):
        super(ISEmodel, self).__init__()
        self.M = num_triplets

        # 21 independent CCC pyramid branches
        self.ccc_modules = nn.ModuleList([CCCPyramid() for _ in range(self.M)])
        # ========== CNN backend branch, corresponding to Table S1 ==========
        self.conv1 = nn.Conv2d(self.M, 2 * self.M, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(2 * self.M)
        self.pool1 = nn.MaxPool2d(kernel_size=4)  # 128 → 32

        self.conv2 = nn.Conv2d(2 * self.M, 4 * self.M, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(4 * self.M)
        self.pool2 = nn.MaxPool2d(kernel_size=4)  # 32 → 8

        self.conv3 = nn.Conv2d(4 * self.M, 4 * self.M, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(4 * self.M)

        # Fully connected layers
        self.fc1 = nn.Linear(4 * self.M * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 36)

    def forward(self, ms_histograms: torch.Tensor):
        # Process each triplet through the CCC pyramid
        ccc_out = []
        for i in range(self.M):
            triplet_hist = ms_histograms[:, i:i + 1, :, :]
            feat = self.ccc_modules[i](triplet_hist)
            ccc_out.append(feat)

        ccc_out = torch.cat(ccc_out, dim=1)
        x = ccc_out.detach()

        # CNN backend forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Flatten + fully connected
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.exp(x)

        return ccc_out, x

# ========== Simple test ==========
if __name__ == "__main__":
    model = ISEmodel(num_triplets=21)
    dummy_input = torch.randn(2, 21, 128, 128)
    ccc_feats, pred_spectrum = model(dummy_input)

    print(f"CCC branch output count: {len(ccc_feats)}")
    print(f"Single CCC output shape: {ccc_feats[0].shape}")  # [2, 1, 128, 128]
    print(f"Predicted spectrum shape: {pred_spectrum.shape}")  # [2, 36]
    print("Model forward propagation is normal")