import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Tuple, List

class BEVLine(nn.Module):
    def __init__(self):
        super(BEVLine, self).__init__()
        self.fusion_layer = ConvFuser(
            in_channels=[80, 256],
            out_channels=256
        )
        self.pts_backbone = SECOND(
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2]
        )
        self.pts_neck = SECONDFPN(
            in_channels=[128, 256],
            out_channels=[256, 256],
            upsample_strides=[1, 2],
            use_conv_for_no_stride=True
        )
        
    def forward(self, x):
        x = self.fusion_layer(x)
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

class ConvFuser(nn.Module):
    def __init__(self, in_channels: List[int] = [80, 256], out_channels: int = 256):
        super().__init__()
        
        # Calculate total input channels (e.g., 80 + 256 = 336)
        self.total_in_channels = sum(in_channels)
        self.out_channels = out_channels
        self.fuser = nn.Sequential(
            nn.Conv2d(self.total_in_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        inputs: List containing [camera_bev_tensor, lidar_bev_tensor]
        """
        # Concatenate along the channel dimension (dim=1)
        x = torch.cat(inputs, dim=1)
        x = self.fuser(x)
        # Fuse the concatenated features
        return x

# reference: https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/backbones/second.py
class SECOND(nn.Module):
    def __init__(self,
                 in_channels: int = 128,
                 out_channels: Sequence[int] = [128, 128, 256],
                 layer_nums: Sequence[int] = [3, 5, 5],
                 layer_strides: Sequence[int] = [2, 2, 2],
                 norm_eps: float = 1e-3,
                 norm_momentum: float = 0.01) -> None:
        super(SECOND, self).__init__()
        
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        # in_filters calculates the input channels for each Stage
        # For example: [128(input), 128(stage1 output), 128(stage2 output)]
        in_filters = [in_channels, *out_channels[:-1]]
        
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            # 1. At the beginning of each Stage: perform spatial downsampling with stride layer_strides[i]
            block = [
                nn.Conv2d(
                    in_channels=in_filters[i],
                    out_channels=out_channels[i],
                    kernel_size=3,
                    stride=layer_strides[i],
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels[i], eps=norm_eps, momentum=norm_momentum),
                nn.ReLU(inplace=True),
            ]
            
            # 2. Stack layer_num basic 3x3 convolution blocks (keeping spatial resolution unchanged)
            for j in range(layer_num):
                block.append(
                    nn.Conv2d(
                        in_channels=out_channels[i],
                        out_channels=out_channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False
                    )
                )
                block.append(nn.BatchNorm2d(out_channels[i], eps=norm_eps, momentum=norm_momentum))
                block.append(nn.ReLU(inplace=True))

            # Pack this Stage into a Sequential module and add it to the overall architecture
            blocks.append(nn.Sequential(*block))

        self.blocks = nn.ModuleList(blocks)

        # Default weight initialization
        self._init_weights()

    def _init_weights(self):
        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x (torch.Tensor): BEV feature map, Shape: (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Output multi-scale feature maps list (corresponding to outputs from different Stages).
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        
        # Return multi-scale feature maps for subsequent feature pyramid fusion via SECONDFPN
        return tuple(outs)

# reference: https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/necks/second_fpn.py
class SECONDFPN(nn.Module):
    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_eps=1e-3,
                 norm_momentum=0.01,
                 use_conv_for_no_stride=False):
        super(SECONDFPN, self).__init__()
        
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            
            # Logic 1: Upsampling (transposed convolution)
            # If stride > 1 or (stride == 1 and force not using regular convolution)
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = nn.ConvTranspose2d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=int(stride),
                    stride=int(stride),
                    bias=False
                )
            # Keep resolution or downsample (using regular convolution)
            # Including cases where stride == 1 and use_conv_for_no_stride=True, including stride < 1 (e.g., 0.5)
            else:
                conv_stride = int(np.round(1 / stride))
                upsample_layer = nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=conv_stride,
                    stride=conv_stride,
                    bias=False
                )

            # Assemble Block: (Conv/Deconv -> BN -> ReLU)
            deblock = nn.Sequential(
                upsample_layer,
                nn.BatchNorm2d(out_channel, eps=norm_eps, momentum=norm_momentum),
                nn.ReLU(inplace=True)
            )
            deblocks.append(deblock)
            
        self.deblocks = nn.ModuleList(deblocks)
        
        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x (List[torch.Tensor]): List/tuple containing multi-scale 4D Tensors (N, C, H, W)
        Returns:
            list[torch.Tensor]: Fused feature maps (wrapped in a list)
        """
        assert len(x) == len(self.in_channels), f"Number of input features {len(x)} does not match the configured {len(self.in_channels)}"
        
        # Pass through corresponding transposed convolution/convolution layers
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        # If multiple feature maps exist, concatenate along channel dimension; otherwise output directly
        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
            
        return [out]
    