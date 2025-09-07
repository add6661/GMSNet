import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.training.GRAF import GRSA
from modules.training.MCRU import Mona

class BasicLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class GMSNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)

        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride=4),
            nn.Conv2d(1, 24, 1, stride=1, padding=0)
        )

        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1),
            BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        )

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )

        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )

        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0)
        )

        self.grsa = GRSA(dim=64, window_size=(8, 8), num_heads=4)

        self.mona = Mona(in_dim=64)

        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

    def _unfold2d(self, x, ws=2):
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(B, C, H // ws, W // ws, ws ** 2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)

    def forward(self, x):
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)

        B, C, H, W = feats.shape
        window_h, window_w = 8, 8
        pad_h = (window_h - H % window_h) % window_h
        pad_w = (window_w - W % window_w) % window_w

        feats = F.pad(feats, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
        _, _, H_pad, W_pad = feats.shape

        feats_reshaped = feats.view(B, C, H_pad // window_h, window_h, W_pad // window_w, window_w)
        feats_reshaped = feats_reshaped.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_h * window_w, C)

        feats_attn = self.grsa(feats_reshaped)

        feats_attn = feats_attn.view(B, H_pad // window_h, W_pad // window_w, window_h, window_w, C)
        feats_attn = feats_attn.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H_pad, W_pad)

        feats = feats_attn[:, :, :H, :W]

        feats_reshape = feats.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, HW, C]
        feats_reshape = self.mona(feats_reshape, hw_shapes=(H, W))
        feats = feats_reshape.transpose(1, 2).reshape(B, C, H, W)  # back to [B, C, H, W]

        heatmap = self.heatmap_head(feats)
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))

        return feats, keypoints, heatmap
