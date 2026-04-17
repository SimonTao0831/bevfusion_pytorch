
import os
import torch
import requests
from bevfusion.bevfusion import BEVFusion

# Load the pure PyTorch BEVFusion model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BEVFusion().to(device)
model.eval()

# Check if the checkpoints directory exists, if not, create it
os.makedirs("checkpoints", exist_ok=True)

# https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth
official_ckpt_path = "checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth" # lidar-cam

# Check if the official checkpoint file exists, if not, download it
if not os.path.exists(official_ckpt_path):
    url = "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
    print(f"Downloading official checkpoint from {url}...")
    response = requests.get(url)
    with open(official_ckpt_path, 'wb') as f:
        f.write(response.content)
    print("Download completed.")
else:
    print(f"Official checkpoint already exists at {official_ckpt_path}.")
    
# Convert and save the checkpoint
output_ckpt_path = "checkpoints/bevfusion_pytorch.pth"
if not os.path.exists(output_ckpt_path):
    print(f"Converting and saving checkpoint to {output_ckpt_path}...")
    model.convert_and_save_checkpoint(model, official_ckpt_path=official_ckpt_path, output_ckpt_path=output_ckpt_path)
else:
    print(f"Checkpoint already exists at {output_ckpt_path}.")
