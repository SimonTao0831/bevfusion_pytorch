import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dModule(nn.Module):
    """Match MMCV ConvModule (Conv2d + BN2d + ReLU) -> weight path: 0.conv, 0.bn"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class Conv1dModule(nn.Module):
    """Match MMCV ConvModule (Conv1d + BN1d + ReLU) -> weight path: 0.conv, 0.bn"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_c, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class PositionEmbedding(nn.Module):
    """Explicit spatial coordinate encoder extracted from official TransFusion"""
    def __init__(self, in_channels=2, embed_dim=128):
        super().__init__()
        # Weight path matches: position_embedding_head.0, .1, .3
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim, kernel_size=1, bias=True),
            nn.BatchNorm1d(embed_dim, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=True)
        )
    def forward(self, x): 
        return self.position_embedding_head(x) # [B, 2, N] -> [B, 128, N]

class SeparateHead(nn.Module):
    def __init__(self, in_channels, heads_dict, head_channels=64):
        super().__init__()
        self.task_names = list(heads_dict.keys())
        for head_name, (out_channels, num_convs) in heads_dict.items():
            # Official implicit setting: intermediate channel defaults to 64.
            self.add_module(head_name, nn.Sequential(
                Conv1dModule(in_channels, head_channels),
                nn.Conv1d(head_channels, out_channels, kernel_size=1, bias=True)
            ))

    def forward(self, x):
        ret = {}
        for head_name in self.task_names:
            ret[head_name] = getattr(self, head_name)(x)
        return ret
    
class TransFusionDecoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=256, dropout=0.1):
        super().__init__()
        # Use ModuleDict to match official naming (e.g., in_proj_weight).
        self.self_attn = nn.ModuleDict({'attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)})
        self.cross_attn = nn.ModuleDict({'attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)})

        # Weight path: ffn.layers.0.0.weight, ffn.layers.1.weight
        self.ffn = nn.ModuleDict({
            'layers': nn.Sequential(
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ),
                nn.Linear(dim_feedforward, d_model)
            )
        })

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

        # Key module that lets the model understand proposal positions.
        self.self_posembed = PositionEmbedding(2, d_model)
        self.cross_posembed = PositionEmbedding(2, d_model)

    def forward(self, query, key, query_pos, key_pos):
        # query/key shape: [B, Seq, C]
        # query_pos/key_pos shape: [B, 2, Seq]
        
        # 1. Inject coordinate-aware positional embedding.
        q_pos_embed = self.self_posembed(query_pos).permute(0, 2, 1) # [B, Seq, C]
        k_pos_embed = self.cross_posembed(key_pos).permute(0, 2, 1)  # [B, Seq, C]

        # 2. Self Attention
        q = query + q_pos_embed
        k = query + q_pos_embed
        v = query
        attn_out, _ = self.self_attn['attn'](q, k, v)
        query = self.norms[0](query + attn_out)

        # 3. Cross Attention (aggregate LiDAR BEV features into queries)
        q = query + q_pos_embed
        k = key + k_pos_embed
        v = key
        attn_out, _ = self.cross_attn['attn'](q, k, v)
        query = self.norms[1](query + attn_out)

        # 4. FFN
        ffn_out = self.ffn['layers'](query)
        query = self.norms[2](query + ffn_out)

        return query

# reference: https://github.com/open-mmlab/mmdetection3d/blob/main/projects/BEVFusion/bevfusion/transfusion_head.py
class TransFusionHead(nn.Module):
    def __init__(
                 self,
                 in_channels=512,
                 hidden_channel=128,
                 num_classes=10,
                 num_proposals=200,
                 nms_kernel_size=3,
                 num_decoder_layers=1,
                 grid_size=[180, 180]): 
        super().__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.hidden_channel = hidden_channel
        self.nms_kernel_size = nms_kernel_size

        # 1. Channel reduction conv (official uses bias conv only, no BN).
        self.shared_conv = nn.Conv2d(in_channels, hidden_channel, kernel_size=3, padding=1, bias=True)

        # 2. Initial heatmap prediction head
        self.heatmap_head = nn.Sequential(
            Conv2dModule(hidden_channel, hidden_channel),
            nn.Conv2d(hidden_channel, num_classes, kernel_size=3, padding=1, bias=True)
        )
        
        # 3. Class encoder
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # 4. Transformer decoder (ModuleList matches decoder.0 path)
        self.decoder = nn.ModuleList([
            TransFusionDecoderLayer(d_model=hidden_channel) for _ in range(num_decoder_layers)
        ])

        # 5. Box regression head
        heads_cfg = {
            'center': (2, 2),
            'height': (1, 2),
            'dim':    (3, 2),
            'rot':    (2, 2),
            'vel':    (2, 2),
            'heatmap':(num_classes, 2) # Required: second-stage scoring mechanism
        }
        # ModuleList matches prediction_heads.0 path
        self.prediction_heads = nn.ModuleList([
            SeparateHead(hidden_channel, heads_cfg) for _ in range(num_decoder_layers)
        ])

        self.register_buffer('bev_pos', self._create_2D_grid(grid_size[0], grid_size[1]), persistent=False)

    def _create_2D_grid(self, x_size, y_size):
        # x_size (H dim) corresponds to LiDAR X axis; y_size (W dim) corresponds to LiDAR Y axis.
        x, y = torch.meshgrid(torch.arange(x_size), torch.arange(y_size), indexing='ij')
        return torch.stack([x.float() + 0.5, y.float() + 0.5], dim=0).view(1, 2, -1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        fusion_feat = self.shared_conv(x)
        fusion_feat_flatten = fusion_feat.view(B, self.hidden_channel, -1) # [B, 128, H*W]
        bev_pos = self.bev_pos.repeat(B, 1, 1) # [B, 2, H*W]

        dense_heatmap = self.heatmap_head(fusion_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        
        padding = self.nms_kernel_size // 2
        local_max = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=padding)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(B, self.num_classes, -1)
        
        top_proposals = heatmap.view(B, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // (H * W)
        top_proposals_index = top_proposals % (H * W)
        
        query_scores = torch.gather(heatmap, dim=-1, index=top_proposals_index.unsqueeze(1).expand(-1, self.num_classes, -1))
        query_feat = torch.gather(fusion_feat_flatten, dim=-1, index=top_proposals_index.unsqueeze(1).expand(-1, self.hidden_channel, -1))
        query_pos = torch.gather(bev_pos, dim=-1, index=top_proposals_index.unsqueeze(1).expand(-1, 2, -1))

        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).transpose(1, 2).float()
        query_feat = query_feat + self.class_encoding(one_hot)

        # Convert shapes for Transformer computation.
        tgt = query_feat.permute(0, 2, 1)     # [B, 200, 128]
        memory = fusion_feat_flatten.permute(0, 2, 1) # [B, H*W, 128]
        
        ret_dicts = []
        for i in range(len(self.decoder)):
            # Official key point: inject query_pos and bev_pos (key_pos).
            tgt = self.decoder[i](query=tgt, key=memory, query_pos=query_pos, key_pos=bev_pos) 
            
            query_out = tgt.permute(0, 2, 1) # [B, 128, 200]
            bbox_preds = self.prediction_heads[i](query_out)
            
            # Add absolute proposal center coordinates.
            bbox_preds['center'] = bbox_preds['center'] + query_pos 
            
            # Update query_pos for the next layer (if num_decoder_layers > 1).
            query_pos = bbox_preds['center'].detach().clone()
            ret_dicts.append(bbox_preds)

        # Return the output from the final decoder layer.
        return tgt, ret_dicts[-1], top_proposals_class, query_scores
