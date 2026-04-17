import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from typing import List
from .ops.voxel.voxelize import Voxelization

class LidarLine(nn.Module):
    def __init__(self):
        super().__init__()
        self.voxelize_reduce = True
        voxelization_op = Voxelization(
            voxel_size=[0.075, 0.075, 0.2],
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            max_num_points=10,
            max_voxels=[120000, 160000]
        )
        self.pts_voxel_layer = voxelization_op
        self.pts_middle_encoder = BEVFusionSparseEncoder(
            in_channels=5,
            sparse_shape=[1440, 1440, 41] # [H, W, D]
        ).cuda()

    def voxelize(self, points: List[torch.Tensor]):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                f, c, n = ret # Hard Voxelize
            else:
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def forward(self, points: List[torch.Tensor]):
        '''
        Args:
            points (List[torch.Tensor]): List of point clouds, each of shape (N_points, 5) where the 5 channels typically represent [X, Y, Z, Intensity, Timestamp/Ring].
        '''
        # Remove padded points
        remove_radius = 1.0
        filtered_points = []
        for p in points:
            mask = (p[:, 0].abs() > remove_radius) | (p[:, 1].abs() > remove_radius)
            filtered_points.append(p[mask])        
        
        feats, coords, sizes = self.voxelize(filtered_points)
        batch_size = coords[-1, 0] + 1
        
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

class SparseBasicBlock(spconv.SparseModule):
    """
    Structure: Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> Add -> ReLU
    """
    def __init__(self, in_channels, out_channels, indice_key=None):
        super().__init__()
        # 1st conv
        self.conv1 = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01)
        
        # 2nd conv
        self.conv2 = spconv.SubMConv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01)

    def forward(self, x):
        assert hasattr(x, "features"), f"SparseBasicBlock expects SparseConvTensor input, got {type(x)!r}"
        identity = x.features  # Save the residual branch
        
        # Conv1 -> BN1 -> ReLU
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(F.relu(out.features, inplace=True))
        
        # Conv2 -> BN2
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        
        # Add -> ReLU
        out = out.replace_feature(out.features + identity)
        out = out.replace_feature(F.relu(out.features, inplace=True))
        
        return out

def make_downsample_layer(in_channels, out_channels, padding, indice_key):
    return spconv.SparseSequential(
        spconv.SparseConv3d(
            in_channels, out_channels, kernel_size=3, stride=2, 
            padding=padding, bias=False, indice_key=indice_key
        ),
        nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01),
        nn.ReLU(inplace=True)
    )

# reference: https://github.com/open-mmlab/mmdetection3d/blob/main/projects/BEVFusion/bevfusion/sparse_encoder.py
class BEVFusionSparseEncoder(nn.Module):
    def __init__(self, in_channels=5, sparse_shape=[1440, 1440, 41]):
        super().__init__()
        self.sparse_shape = sparse_shape
        
        # 1. Initial input layer: pts_middle_encoder.conv_input
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, 16, kernel_size=3, padding=1, bias=False, indice_key='subm1'),
            nn.BatchNorm1d(16, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        # 2. Core encoder layers: pts_middle_encoder.encoder_layers
        self.encoder_layers = nn.ModuleDict({
            
            # Layer 1: 16 -> 16 -> 32 (Downsample)
            'encoder_layer1': spconv.SparseSequential(
                SparseBasicBlock(16, 16, indice_key='subm1'),
                SparseBasicBlock(16, 16, indice_key='subm1'),
                make_downsample_layer(16, 32, padding=1, indice_key='spconv1')
            ),
            
            # Layer 2: 32 -> 32 -> 64 (Downsample)
            'encoder_layer2': spconv.SparseSequential(
                SparseBasicBlock(32, 32, indice_key='subm2'),
                SparseBasicBlock(32, 32, indice_key='subm2'),
                make_downsample_layer(32, 64, padding=1, indice_key='spconv2')
            ),
            
            # Layer 3: 64 -> 64 -> 128 (Downsample)
            # Note: padding=(1, 1, 0) is designed specifically for Z-axis boundary alignment
            'encoder_layer3': spconv.SparseSequential(
                SparseBasicBlock(64, 64, indice_key='subm3'),
                SparseBasicBlock(64, 64, indice_key='subm3'),
                make_downsample_layer(64, 128, padding=(1, 1, 0), indice_key='spconv3')
            ),
            
            # Layer 4: 128 -> 128 (no downsample)
            'encoder_layer4': spconv.SparseSequential(
                SparseBasicBlock(128, 128, indice_key='subm4'),
                SparseBasicBlock(128, 128, indice_key='subm4')
            )
        })
        
        # 3. Output layer (BEVFusion-specific Z-axis compression): pts_middle_encoder.conv_out
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(
                128, 128, kernel_size=(1, 1, 3), stride=(1, 1, 2), 
                padding=0, bias=False, indice_key='spconv_down2'
            ),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

    def forward(self, voxel_features, coors, batch_size):
        """
        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Return spatial features in shape (N, C*D, H, W).
        """
        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        
        # 1. Input
        x = self.conv_input(x)
        
        # 2. Encoder Layers
        x = self.encoder_layers['encoder_layer1'](x)
        x = self.encoder_layers['encoder_layer2'](x)
        x = self.encoder_layers['encoder_layer3'](x)
        x = self.encoder_layers['encoder_layer4'](x)
        
        # 3. Output
        out = self.conv_out(x)
        
        # 4. Dense projection to BEV space
        spatial_features = out.dense()
        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)
        
        return spatial_features
