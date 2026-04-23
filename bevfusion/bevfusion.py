import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from .lidar_line import LidarLine
from .camera_line import CameraLine
from .bev_line import BEVLine
from .bev_head import TransFusionHead

class BEVFusion(nn.Module):
    # Detection parameters
    PC_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    VOXEL_SIZE = [0.075, 0.075]
    OUT_SIZE_FACTOR = 8
    
    # Visualization parameters
    VIZ_XLIM = (-54, 54)
    VIZ_YLIM = (-54, 54)
    VIZ_FIGSIZE = (12, 12)
    CMAP_NUM_COLORS = 20
    NUSCENES_CLASS_NAMES = {
        0: 'car',
        1: 'truck',
        2: 'construction_vehicle',
        3: 'bus',
        4: 'trailer',
        5: 'barrier',
        6: 'motorcycle',
        7: 'bicycle',
        8: 'pedestrian',
        9: 'traffic_cone',
    }
    
    # Swin backbone weight mapping
    SWIN_LAYER_MAPPING = {
        'img_backbone.stages.0.blocks': 'img_backbone.swin.features.1',
        'img_backbone.stages.1.blocks': 'img_backbone.swin.features.3',
        'img_backbone.stages.2.blocks': 'img_backbone.swin.features.5',
        'img_backbone.stages.3.blocks': 'img_backbone.swin.features.7',
        'img_backbone.stages.0.downsample': 'img_backbone.swin.features.2',
        'img_backbone.stages.1.downsample': 'img_backbone.swin.features.4',
        'img_backbone.stages.2.downsample': 'img_backbone.swin.features.6',
    }
    SWIN_PATCH_MAPPING = {
        'img_backbone.patch_embed.projection': 'img_backbone.swin.features.0.0',
        'img_backbone.patch_embed.norm': 'img_backbone.swin.features.0.2',
    }
    SWIN_COMPONENT_MAPPING = {
        '.attn.w_msa.': '.attn.',
        '.ffn.layers.0.0': '.mlp.0',
        '.ffn.layers.1': '.mlp.3',
    }
    
    def __init__(self):
        super(BEVFusion, self).__init__()
        self.lidar_line = LidarLine()
        self.camera_line = CameraLine()
        self.bev_line = BEVLine()
        self.bbox_head = TransFusionHead()
        
    def forward(self, imgs, points_mm, mats_dict_mm):
        # Extract features from camera and lidar branches
        cam_feats = self.camera_line(imgs, points_mm, mats_dict_mm)
        lidar_feats = self.lidar_line(points_mm)
        
        # Fuse features in BEV space
        features = [cam_feats, lidar_feats]
        bev_feats = self.bev_line(features)
        
        # Generate detections
        tgt, bbox_preds, classes, scores = self.bbox_head(bev_feats)
        
        return {
            "lidar_feats": lidar_feats,
            "cam_feats": cam_feats,
            "bev_feats": bev_feats,
            "tgt": tgt,
            "bbox_preds": bbox_preds,
            "classes": classes,
            "scores": scores
        }

    def _map_backbone_keys(self, key):
        """Apply Swin backbone weight mapping."""
        # Apply patch embedding mapping
        for old, new in self.SWIN_PATCH_MAPPING.items():
            if old in key:
                key = key.replace(old, new)
        
        # Apply layer mapping
        for old, new in self.SWIN_LAYER_MAPPING.items():
            if old in key:
                key = key.replace(old, new)
        
        # Apply component mapping
        for old, new in self.SWIN_COMPONENT_MAPPING.items():
            key = key.replace(old, new)
        
        return key
    
    def _remap_checkpoint_keys(self, old_state_dict):
        """Remap checkpoint keys from official model to current architecture."""
        new_state_dict = {}
        
        for key, value in old_state_dict.items():
            # Skip relative position indices
            if 'relative_position_index' in key:
                continue
            
            new_key = key
            
            # A. Camera line
            if key.startswith(('img_backbone.', 'img_neck.', 'view_transform.')):
                if key.startswith('img_backbone.'):
                    new_key = self._map_backbone_keys(key)
                elif key.startswith('img_neck.'):
                    new_key = key.replace('.conv.', '.0.').replace('.bn.', '.1.')
                
                new_key = f"camera_line.{new_key}"
            
            # B. LiDAR line
            elif key.startswith('pts_middle_encoder.'):
                new_key = f"lidar_line.{key}"
            
            # C. BEV line
            elif key.startswith('fusion_layer.'):
                new_key = key.replace('fusion_layer.', 'bev_line.fusion_layer.fuser.')
            elif key.startswith(('pts_backbone.', 'pts_neck.')):
                new_key = f"bev_line.{key}"
            
            # D. Detection head (no mapping needed)
            elif key.startswith('bbox_head.'):
                new_key = key
            
            new_state_dict[new_key] = value
        
        return new_state_dict
    
    def convert_and_save_checkpoint(self, model, official_ckpt_path, output_ckpt_path):
        """Load checkpoint from official model and save with current architecture keys."""
        checkpoint = torch.load(official_ckpt_path, map_location='cpu', weights_only=False)
        old_state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Remap keys
        new_state_dict = self._remap_checkpoint_keys(old_state_dict)
        
        # Load into model
        msg = model.load_state_dict(new_state_dict, strict=False)
        
        # Check for missing core weights
        ignored_substrings = [
            'relative_position_index',  # 2D static relative position index for Swin
            'camera_line.mean',         # Mean buffer for camera input
            'camera_line.std',          # Standard deviation buffer for camera input
            'swin.head.',               # Image classification head left over from Swin pre-training
            'swin.norm.'                # Global normalization layer from Swin pre-training (note the trailing dot)
        ]
        real_missing = [
            k for k in msg.missing_keys 
            if all(x not in k for x in ignored_substrings)
        ]
        
        if not real_missing:
            print("Core weights matched perfectly!")
        else:
            print(f"Warning: Some core weights missing: {real_missing}")
        
        # Save converted checkpoint
        torch.save(model.state_dict(), output_ckpt_path)
    
    @staticmethod
    def decode_bbox(bbox_preds, score_threshold=0.2):
        """
        Decode bounding box predictions from network output.
        
        Args:
            bbox_preds: Dictionary containing detection predictions with keys:
                - heatmap: Class scores [B, C, H*W]
                - center: Box center coordinates [B, 2, H*W]
                - height: Box height [B, 1, H*W]
                - dim: Box dimensions (length, width) [B, 2, H*W]
                - rot: Box rotation (sin, cos) [B, 2, H*W]
            score_threshold: Minimum confidence score for detections (default: 0.2)
        
        Returns:
            List of detection results per batch containing:
                - boxes: [N, 7] array with (x, y, z, l, w, h, yaw)
                - scores: [N] array of confidence scores
                - labels: [N] array of class labels
        """
        # Extract and process classification scores
        scores = bbox_preds['heatmap'].sigmoid()
        max_scores, labels = torch.max(scores, dim=1)  # [B, H*W]
        
        # Decode center coordinates from grid to real world
        center = bbox_preds['center'].clone()
        center[:, 0, :] = center[:, 0, :] * BEVFusion.VOXEL_SIZE[0] * BEVFusion.OUT_SIZE_FACTOR + BEVFusion.PC_RANGE[0]
        center[:, 1, :] = center[:, 1, :] * BEVFusion.VOXEL_SIZE[1] * BEVFusion.OUT_SIZE_FACTOR + BEVFusion.PC_RANGE[1]
        
        # Decode box dimensions
        dims = torch.exp(bbox_preds['dim'])
        
        # Decode yaw angle
        rot = torch.atan2(bbox_preds['rot'][:, 0, :], bbox_preds['rot'][:, 1, :])
        
        # Package results
        final_boxes = []
        for b in range(scores.shape[0]):
            mask = max_scores[b] > score_threshold
            batch_boxes = torch.cat([
                center[b, :, mask].T,
                bbox_preds['height'][b, :, mask].T,
                dims[b, :, mask].T,
                rot[b, mask].unsqueeze(-1)
            ], dim=-1)
            
            final_boxes.append({
                'boxes': batch_boxes.detach().cpu().numpy(),
                'scores': max_scores[b, mask].detach().cpu().numpy(),
                'labels': labels[b, mask].detach().cpu().numpy()
            })
        
        return final_boxes
    
    @staticmethod
    def _compute_box_corners(x, y, length, width, yaw):
        """
        Compute 2D bounding box corners in BEV.
        
        Args:
            x, y: Box center coordinates
            length, width: Box dimensions
            yaw: Box heading angle in radians
        
        Returns:
            corners: [5, 2] array of corner coordinates (closed path for plotting)
        """
        half_l = length * 0.5
        half_w = width * 0.5
        
        # Local box corners
        local_corners = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
            [half_l, half_w],  # Close the loop
        ], dtype=np.float32)
        
        # Rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
        
        # Transform to world coordinates
        world_corners = local_corners @ rot_matrix.T
        world_corners[:, 0] += x
        world_corners[:, 1] += y
        
        return world_corners
    
    def visualize_results(self, points, bbox_results, score_thr=0.3, point_size=0.2, point_alpha=0.35, point_color='gray'):
        """
        Visualize BEV detection results with point cloud and bounding boxes.
        
        Args:
            points: Point cloud [N, >=3] or [1, N, >=3]
            bbox_results: Detection results from decode_bbox
            score_thr: Score threshold for filtering detections
            point_size: Size of point cloud markers
            point_alpha: Alpha transparency of point cloud
            point_color: Color of point cloud
        """
        # Prepare point cloud
        pts = np.asarray(points)
        if pts.ndim == 3:
            pts = pts[0]
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        
        # Extract detection results
        result = bbox_results[0]
        boxes = result['boxes'].copy()
        scores = result['scores']
        labels = result['labels']
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.VIZ_FIGSIZE)
        
        # Draw point cloud background
        ax.scatter(pts[:, 0], pts[:, 1], s=point_size, c=point_color, alpha=point_alpha)
        
        # Get colormap for class differentiation
        cmap = plt.get_cmap('tab20')
        
        # Draw detections
        for i in range(len(boxes)):
            if scores[i] < score_thr:
                continue
            
            x, y, z, length, width, height, yaw = boxes[i]
            class_id = int(labels[i])
            class_name = self.NUSCENES_CLASS_NAMES.get(class_id, f'Unknown_{class_id}')
            
            box_color = cmap(class_id % self.CMAP_NUM_COLORS)
            
            # Draw bounding box
            corners = self._compute_box_corners(x, y, length, width, yaw)
            ax.plot(corners[:, 0], corners[:, 1], color=box_color, linewidth=1.8)
            
            # Draw heading arrow
            arrow_length = max(length, width) * 0.6
            ax.arrow(
                x, y,
                np.cos(yaw) * arrow_length,
                np.sin(yaw) * arrow_length,
                color=box_color,
                width=0.08,
                head_width=0.6,
                head_length=0.9,
                length_includes_head=True,
            )
            
            # Draw label
            ax.text(
                x, y,
                f'{class_name} {scores[i]:.2f}',
                color='white',
                fontsize=8,
                bbox=dict(facecolor=box_color, alpha=0.8, edgecolor='none', pad=1.5),
            )
        
        # Configure plot
        ax.set_xlim(self.VIZ_XLIM)
        ax.set_ylim(self.VIZ_YLIM)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.4)
        ax.set_xlabel('Back <--> Front')
        ax.set_ylabel('Right <--> Left')
        ax.set_title('BEVfusion Detection Result')
        
        plt.show()
