import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t
from .ops.bev_pool.bev_pool import bev_pool

class CameraLine(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1))
        self.img_backbone = SwinTransformer()
        self.img_neck = GeneralizedLSSFPN(in_channels=[192, 384, 768], out_channels=256, start_level=0)
        self.view_transform = DepthLSSTransform(
            in_channels=256,
            out_channels=80,
            image_size=[256, 704],
            feature_size=[32, 88],
            xbound=[-54.0, 54.0, 0.3],
            ybound=[-54.0, 54.0, 0.3],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 0.5],
            downsample=2
        )
        
    def forward(self, imgs, points, mats_dict):
        """
        Args:
            imgs (torch.Tensor): Input tensor of shape (B, 6, 3, H, W)
            mats_dict (dict): Dictionary containing transformation matrices
            points (torch.Tensor, optional): Input tensor of shape (B, N, 3)

        Returns:
            Tensor: Output tensor of shape (B, 80, 180, 180)
        """
        B, N, C, H, W = imgs.size()
        x = (imgs.float() - self.mean) / self.std
        x = x.view(B * N, C, H, W).contiguous()
        
        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]
                    
        BN, C_feat, H_feat, W_feat = x.size()
        x = x.view(B, int(BN / B), C_feat, H_feat, W_feat)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                mats_dict['lidar2image'],
                mats_dict['camera_intrinsics'],
                mats_dict['camera2lidar'],
                mats_dict['img_aug_matrix'],
                mats_dict['lidar_aug_matrix']
            )
        return x

class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load the Swin-Tiny, default config: embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=[7, 7], stochastic_depth_prob=0.2
        # checkpoint: https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
        self.swin = swin_t(weights=None)
        self.norm1 = nn.LayerNorm(192)
        self.norm2 = nn.LayerNorm(384)
        self.norm3 = nn.LayerNorm(768)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            List[torch.Tensor]: A list of three tensors of shape (B, C, H, W) corresponding to the output of each stage.
            (B, 192, H/8, W/8), (B, 384, H/16, W/16), (B, 768, H/32, W/32)
        """
        # Stage 0 (96 channels)
        x = self.swin.features[0](x)
        x = self.swin.features[1](x)
        x = self.swin.features[2](x)
        
        # Stage 1 (192 channels) -> out_index 1
        x = self.swin.features[3](x)
        out1 = self.norm1(x).permute(0, 3, 1, 2).contiguous() 
        x = self.swin.features[4](x)
        
        # Stage 2 (384 channels) -> out_index 2
        x = self.swin.features[5](x)
        out2 = self.norm2(x).permute(0, 3, 1, 2).contiguous()
        x = self.swin.features[6](x)
        
        # Stage 3 (768 channels) -> out_index 3
        x = self.swin.features[7](x)
        out3 = self.norm3(x).permute(0, 3, 1, 2).contiguous()
        
        return [out1, out2, out3]

# reference: https://github.com/mit-han-lab/bevfusion/blob/main/mmdet3d/models/necks/generalized_lss.py
class GeneralizedLSSFPN(nn.Module):
    def __init__(
        self,
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        
        self.backbone_end_level = len(in_channels) - 1 # 2
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # Build the dynamic lateral and fpn convolutions
        for i in range(self.start_level, self.backbone_end_level): # 0, 1
            # For the deepest connection: channel[i] + channel[i+1] (384 + 768 = 1152)
            # For shallower connections: channel[i] + out_channels  (192 + 256 = 448)
            if i == self.backbone_end_level - 1:
                lat_in_channels = in_channels[i] + in_channels[i + 1]
            else:
                lat_in_channels = in_channels[i] + out_channels

            # 1x1 lateral
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(lat_in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

            # 3x3 FPN smooth
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): List of input feature maps
            inputs[0]: (B, 192, H/8, W/8)
            inputs[1]: (B, 384, H/16, W/16)
            inputs[2]: (B, 768, H/32, W/32)

        Returns:
            Tensor: Output feature map of shape (N, C, H, W)
            outs[0]: (B, 256, H/8, W/8)
        """
        assert len(inputs) == len(self.in_channels)

        # Isolate the levels we actually want to process
        laterals = [inputs[i + self.start_level] for i in range(len(inputs) - self.start_level)]
        used_backbone_levels = len(laterals) - 1

        # Build top-down path
        for i in range(used_backbone_levels - 1, -1, -1): # 0, 1 (reverse order)
            
            # Upsample the deeper feature map
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode="bilinear",
                align_corners=False
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # Build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        
        return outs[0]
    
# reference: https://github.com/open-mmlab/mmdetection3d/blob/main/projects/BEVFusion/bevfusion/depth_lss.py
def gen_dx_bx(xbound, ybound, zbound):
    # BEV map: 360 x 360 x 1, cell size: 0.3m
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]) # [0.3, 0.3, 20.0]
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]) # [-53.85, -53.85, 0.0]
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]) # [360, 360, 1]
    return dx, bx, nx

class DepthLSSTransform(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.C = out_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.dbound = dbound

        # Generate and register coordinate parameters (aligned with state_dict: view_transform.dx, bx, nx)
        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        # Generate frustum parameters (view_transform.frustum)
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]  # 118 depth bins

        # 1. Depth map feature extraction network (with bias)
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, kernel_size=5, stride=4, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 2. Core depth prediction network (with bias)
        # Input: 256 (image) + 64 (depth map) = 320
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, self.D + self.C, kernel_size=1, bias=True)
        )

        # 3. BEV downsampling network (without bias)
        if downsample == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=downsample, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.downsample = nn.Identity()

    def create_frustum(self):
        iH, iW = self.image_size # [256, 704]
        fH, fW = self.feature_size # [32, 88] => downsample by 8x from image size

        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW) # [118] => [118, 1, 1] => [118, 32, 88]
        D, _, _ = ds.shape # (60 - 1) x 2 = 118
        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        
        frustum = torch.stack((xs, ys, ds), -1) # [118, 32, 88, 3]
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans, extra_rots, extra_trans):
        B, N, _ = camera2lidar_trans.shape

        # Undo post-transformation
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        
        # Camera to LiDAR
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        # LiDAR Augmentations
        points = extra_rots.view(B, 1, 1, 1, 1, 3, 3).repeat(1, N, 1, 1, 1, 1, 1).matmul(points.unsqueeze(-1)).squeeze(-1)
        points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def forward(self, img_feats, points, lidar2image, cam_intrinsic, camera2lidar, img_aug_matrix, lidar_aug_matrix):
        """
        Args:
            img_feats: (B, N, 256, 32, 88)
            points: (B, N, 3)
            lidar2image: (B, N, 4, 4)
            cam_intrinsic: (B, N, 4, 4)
            camera2lidar: (B, N, 4, 4)
            img_aug_matrix: (B, N, 4, 4)
            lidar_aug_matrix: (B, N, 4, 4)
        Returns:
            final: (B, 80, 180, 180)
        """
        B, N, C, fH, fW = img_feats.shape
        
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        # 1. Project LiDAR point clouds into images to build sparse depth maps
        depth_map = torch.zeros(B, N, 1, *self.image_size).to(img_feats.device) # (B, N, 1, 256, 704)

        for b in range(B):
            cur_coords = points[b][:, :3].clone()
            
            # Undo LiDAR Aug -> Lidar2Image -> Depth
            cur_coords -= lidar_aug_matrix[b][:3, 3]
            cur_coords = torch.inverse(lidar_aug_matrix[b][:3, :3]).matmul(cur_coords.transpose(1, 0))
            cur_coords = lidar2image[b][:, :3, :3].matmul(cur_coords)
            cur_coords += lidar2image[b][:, :3, 3].reshape(-1, 3, 1)
            
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # Apply Image Augmentation
            cur_coords = img_aug_matrix[b][:, :3, :3].matmul(cur_coords)
            cur_coords += img_aug_matrix[b][:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)
            cur_coords = cur_coords[..., [1, 0]] # XY to YX

            on_img = ((cur_coords[..., 0] < self.image_size[0]) & (cur_coords[..., 0] >= 0) &
                      (cur_coords[..., 1] < self.image_size[1]) & (cur_coords[..., 1] >= 0))
            
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth_map[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        # 2. Extract features and fuse them with depth maps
        d = depth_map.view(B * N, *depth_map.shape[2:])
        x = img_feats.view(B * N, C, fH, fW)
        
        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        # Split depth distribution and context features
        depth_probs = x[:, :self.D].softmax(dim=1)
        x = depth_probs.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        x = x.view(B, N, self.C, self.D, fH, fW).permute(0, 1, 3, 4, 5, 2)

        # 3. Compute 3D geometry coordinates
        geom = self.get_geometry(
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans,
            extra_rots=lidar_aug_matrix[..., :3, :3], extra_trans=lidar_aug_matrix[..., :3, 3]
        )

        # 4. Run CUDA BEV pooling
        Nprime = B * N * self.D * fH * fW
        x = x.reshape(Nprime, self.C)
        
        geom_feats = ((geom - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        kept = ((geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) &
                (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) &
                (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2]))
        
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        # Core aggregation step
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = torch.cat(x.unbind(dim=2), 1)
        
        # 5. Final downsampling
        final = self.downsample(final)

        return final
    