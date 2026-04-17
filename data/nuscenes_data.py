# Reference: https://github.com/E2E-AD/AD-MLP/blob/main/deps/stp3/stp3/datas/NuscenesData.py
import os
import numpy as np
from PIL import Image

from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class NuscenesData(Dataset):
    def __init__(self, nusc, is_train, pre_frames, future_frames):
        self.nusc = nusc
        self.is_train = is_train # 0: train, 1: val (2: test)

        self.pre_frames = pre_frames + 1 # previous frames + current frame
        self.future_frames = future_frames
        self.sequence_length = self.pre_frames + self.future_frames # previous frames + future framess
        
        self.max_lidar_points = 35000
        self.front_fov = 77
        self.cameras = ["CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"] # (H, W, 3): (900, 1600, 3) <class 'numpy.ndarray'>

        self.can_bus = NuScenesCanBus(dataroot=self.nusc.dataroot)

        self.scenes = self.get_scenes()
        self.samples = self.get_samples()
        self.indices = self.get_indices()
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data = {}
        keys = ['token', 'raw_images', 'raw_lidar', 'lidar_ego', 'fused_pc', 'future_waypoints', 
                'cur_waypoint', 'pre_waypoints', 'instance', 'velocity', 'accel', 
                'yaw_rate', 'command', 'image_paths', 'lidar_path', 'lidar_cs_record',
                'lidar2image', 'cam_intrinsic', 'camera2lidar']
        for key in keys:
            data[key] = []

        cur_frame = self.indices[index][self.pre_frames - 1] # current frame
        cur_sample = self.samples[cur_frame] # current sample
        T_ego2w_cur = self.get_transform_w2ego(cur_sample, True) # current transform matrix, ego -> world
        data['token'].append(cur_sample['token'])

        for idx, frame in enumerate(self.indices[index]):
            sample = self.samples[frame]
            # Get trainning data
            if idx < self.pre_frames: # previous frames + current frame
                # Camera images
                images, image_paths = self.get_images(sample, self.cameras) # [num_cameras, 3, H, W]
                images = images.unsqueeze(0) # [1, num_cameras, 3, H, W]
                data['raw_images'].append(images)
                data['image_paths'].append(image_paths)

                # Lidar points
                lidar_ego, raw_lidar, lidar_path, lidar_cs_record = self.get_lidar(sample)
                fused_pc = self.get_fused_lidar_sweeps(sample)
                lidar_ego = lidar_ego.unsqueeze(0) # [1, N, 4], in ego coord system
                raw_lidar = raw_lidar.unsqueeze(0) # [1, N, 4], in lidar coord system
                data['lidar_ego'].append(lidar_ego)
                data['raw_lidar'].append(raw_lidar)
                data['fused_pc'].append(fused_pc)
                data['lidar_path'].append(lidar_path)
                data['lidar_cs_record'].append(lidar_cs_record)
                
                # previous waypoints
                pre_waypoints = self.get_waypoints(sample, T_ego2w_cur).float() # [1, waypoint] 
                data['pre_waypoints'].append(pre_waypoints)

                # Record ego states
                instance = self.get_instance(sample)
                data['instance'].append(instance) # list[T][B]
                velocity, accel, yaw_rate = self.get_ego_state(sample)
                data['velocity'].append(velocity)
                data['accel'].append(accel)
                data['yaw_rate'].append(yaw_rate)
                
                # Transformation matrices
                lidar2image, cam_intrinsic, camera2lidar = self.get_camera_mats(sample, self.cameras)
                data['lidar2image'].append(lidar2image)
                data['cam_intrinsic'].append(cam_intrinsic)
                data['camera2lidar'].append(camera2lidar)
            else:
                # future waypoints
                future_waypoints = self.get_waypoints(sample, T_ego2w_cur).float() # [1, waypoint]
                data['future_waypoints'].append(future_waypoints)
        
        data['raw_images'] = torch.cat(data['raw_images'], dim=0) # [batch_size, pre_frames, num_cameras, 3, H, W]
        data['raw_lidar'] = torch.cat(data['raw_lidar'], dim=0) # [pre_frames, max_N, 4], in lidar coord system
        data['lidar_ego'] = torch.cat(data['lidar_ego'], dim=0) # [pre_frames, max_N, 4], in ego coord system
        data['fused_pc'] = torch.stack(data['fused_pc'], dim=0) # [pre_frames, max_N, 4], in lidar coord system
        data['lidar2image'] = torch.stack(data['lidar2image'], dim=0) # [pre_frames, num_cameras, 4, 4]
        data['cam_intrinsic'] = torch.stack(data['cam_intrinsic'], dim=0) # [pre_frames, num_cameras, 4, 4]
        data['camera2lidar'] = torch.stack(data['camera2lidar'], dim=0) # [pre_frames, num_cameras, 4, 4]
        data['pre_waypoints'] = torch.cat(data['pre_waypoints'], dim=0) # [batch_size, pre_frames, waypoint]
        data['cur_waypoint'] = data['pre_waypoints'][-1].unsqueeze(0) # the last previous waypoint is the current waypoint
        data['pre_waypoints'] = data['pre_waypoints'][:-1] # remove the current waypoint
        data['velocity'] = torch.stack(data['velocity'], dim=0)  # [pre_frames, 2]
        data['accel'] = torch.stack(data['accel'], dim=0)  # [pre_frames, 2]
        data['yaw_rate'] = torch.stack(data['yaw_rate'], dim=0)  # [pre_frames]
        if self.future_frames != 0:
            data['future_waypoints'] = torch.cat(data['future_waypoints'], dim=0) # [batch_size, future_frames, waypoint]
            data['command'] = self.classify_command(data['future_waypoints']) # [batch_size, command]   
        return data
    
    # Get splits scenes
    def get_scenes(self):
        # filter by scene split
        split = {'v1.0-trainval': {0: 'train', 1: 'val', 2: 'test'},
                 'v1.0-mini': {0: 'mini_train', 1: 'mini_val'},}[self.nusc.version][self.is_train]

        blacklist = [419] + self.can_bus.can_blacklist  # scene-0419 does not have vehicle monitor data
        blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]

        scenes = create_splits_scenes()[split][:]
        for scene_no in blacklist:
            if scene_no in scenes:
                scenes.remove(scene_no)

        return scenes

    # Get samples from splits scenes and sort them
    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    # Get frames index in same scene
    def get_indices(self):
        indices = []
        for index in range(len(self.samples)):
            is_valid_data = True
            previous_sample = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.samples):
                    is_valid_data = False
                    break
                sample = self.samples[index_t]
                # Check if scene is the same
                if (previous_sample is not None) and (sample['scene_token'] != previous_sample['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_sample = sample

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)
    
    def get_images(self, sample, cameras):
        images = []
        image_paths = []

        for cam in cameras:
            cam_token = self.nusc.get('sample_data', sample['data'][cam])
            image_path = self.nusc.get_sample_data_path(cam_token['token'])
            relative_path = os.path.relpath(image_path, self.nusc.dataroot)
            image_paths.append(relative_path)

            img = np.array(Image.open(image_path))
            img_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2) # [1, H, W, 3] -> [1, 3, H, W]
            images.append(img_tensor)

        images = torch.cat(images, dim=0) # [cam_num, 3, H, W]
        return images, image_paths

    def get_camera_mats(self, sample, cameras=None):
        """
        Args:
            sample (dict): nuScenes sample record.
            cameras (list[str], optional): Camera names. Defaults to self.cameras.

        Returns:
            lidar2image (torch.Tensor): [num_cameras, 4, 4]
            cam_intrinsic (torch.Tensor): [num_cameras, 4, 4]
            camera2lidar (torch.Tensor): [num_cameras, 4, 4]
        """
        if cameras is None:
            cameras = self.cameras

        def invert_rigid(transform):
            inv = np.eye(4, dtype=np.float32)
            rot = transform[:3, :3]
            trans = transform[:3, 3]
            inv[:3, :3] = rot.T
            inv[:3, 3] = -(rot.T @ trans)
            return inv

        lidar_sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])

        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = Quaternion(lidar_cs_record['rotation']).rotation_matrix.astype(np.float32)
        lidar2ego[:3, 3] = np.asarray(lidar_cs_record['translation'], dtype=np.float32)
        ego2lidar = invert_rigid(lidar2ego)

        lidar2image_list = []
        cam_intrinsic_list = []
        camera2lidar_list = []

        for cam in cameras:
            cam_sample_data = self.nusc.get('sample_data', sample['data'][cam])
            cam_cs_record = self.nusc.get('calibrated_sensor', cam_sample_data['calibrated_sensor_token'])

            # nuScenes calibrated sensor pose is sensor -> ego.
            camera2ego = np.eye(4, dtype=np.float32)
            camera2ego[:3, :3] = Quaternion(cam_cs_record['rotation']).rotation_matrix.astype(np.float32)
            camera2ego[:3, 3] = np.asarray(cam_cs_record['translation'], dtype=np.float32)
            ego2camera = invert_rigid(camera2ego)

            # True camera -> lidar transform.
            camera2lidar = ego2lidar @ camera2ego

            # Input points are assumed to be in lidar frame.
            points2camera = ego2camera @ lidar2ego

            intrinsic = np.eye(4, dtype=np.float32)
            intrinsic[:3, :3] = np.asarray(cam_cs_record['camera_intrinsic'], dtype=np.float32)

            lidar2image_list.append(intrinsic @ points2camera)
            cam_intrinsic_list.append(intrinsic)
            camera2lidar_list.append(camera2lidar)

        lidar2image = torch.from_numpy(np.stack(lidar2image_list, axis=0)).float()
        cam_intrinsic = torch.from_numpy(np.stack(cam_intrinsic_list, axis=0)).float()
        camera2lidar = torch.from_numpy(np.stack(camera2lidar_list, axis=0)).float()

        return lidar2image, cam_intrinsic, camera2lidar
    
    def get_lidar(self, sample):
        target_num_points = self.max_lidar_points
        lidar_token = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', lidar_token['calibrated_sensor_token'])
        lidar_path = self.nusc.get_sample_data_path(lidar_token['token'])
        relative_path = os.path.relpath(lidar_path, self.nusc.dataroot)
        lidar = LidarPointCloud.from_file(lidar_path).points.T
        lidar = torch.tensor(lidar, dtype=torch.float32) # [N, 4]

        # lidar -> ego
        points = lidar[:, :3]  # [N, 3]
        # translation vector
        t = torch.tensor(cs_record['translation'], dtype=torch.float32)  # [3]
        # rotation quaternion
        q = Quaternion(cs_record['rotation'])
        r = torch.tensor(q.rotation_matrix, dtype=torch.float32)
        points_ego = r @ points.T + t.view(3, 1)  # [3, N]
        points_ego = points_ego.T  # [N, 3]

        lidar_ego = torch.cat([points_ego, lidar[:, 3:4]], dim=1)  # [N, 4]

        total_num_points = lidar_ego.size(0)
        # Pad lidar to ensure consistent tensor dimensions
        if total_num_points < target_num_points:
            padding = (0, 0, 0, target_num_points - total_num_points)
            lidar_ego = F.pad(lidar_ego, padding, value=0)
            raw_lidar = F.pad(lidar, padding, value=0)
        else:
            sampled_indices = farthest_point_sampling(lidar_ego[:, :3], target_num_points)
            lidar_ego = lidar_ego[sampled_indices]
            raw_lidar = lidar[sampled_indices]
        return lidar_ego, raw_lidar, relative_path, cs_record # [max_N, 4]

    def get_fused_lidar_sweeps(self, sample, sweeps_num=9):
        def get_matrices(rec):
            t = torch.tensor(rec['translation'], dtype=torch.float32).view(3, 1)
            r = torch.tensor(Quaternion(rec['rotation']).rotation_matrix, dtype=torch.float32)
            return r, t

        target_num_points = self.max_lidar_points * (sweeps_num + 1)

        # 1. Get metadata of the reference frame (current key frame).
        ref_lidar_token = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ref_time = ref_lidar_token['timestamp']
        
        # Reference frame extrinsics (cs) and pose (ego).
        ref_cs_rec = self.nusc.get('calibrated_sensor', ref_lidar_token['calibrated_sensor_token'])
        ref_pose_rec = self.nusc.get('ego_pose', ref_lidar_token['ego_pose_token'])
        
        ref_cs_r, ref_cs_t = get_matrices(ref_cs_rec)
        ref_pose_r, ref_pose_t = get_matrices(ref_pose_rec)

        all_points_5d = []
        curr_sd_token = ref_lidar_token['token']
        last_pc_5d = None
        
        # 2. Iteratively trace back to collect sweeps.
        for i in range(sweeps_num + 1):
            if not curr_sd_token:
                break
            
            sd_record = self.nusc.get('sample_data', curr_sd_token)
            
            # Load point cloud [N, 4].
            lidar_path = self.nusc.get_sample_data_path(sd_record['token'])
            pc = LidarPointCloud.from_file(lidar_path)
            points = torch.tensor(pc.points.T, dtype=torch.float32) # [N, 4] (X, Y, Z, I)
            mask_close = (torch.abs(points[:, 0]) < 1.5) & (torch.abs(points[:, 1]) < 1.8)
            points = points[~mask_close]
            
            # Compute positive time delta in seconds.
            time_lag = torch.tensor([(ref_time - sd_record['timestamp']) / 1e6], dtype=torch.float32)
            
            xyz = points[:, :3] # [N, 3]
            intensity = points[:, 3:4] # [N, 1]

            if i == 0:
                # Use current frame directly.
                points_fused = xyz
            else:
                # Spatial compensation for historical frames.
                curr_cs_r, curr_cs_t = get_matrices(self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token']))
                curr_pose_r, curr_pose_t = get_matrices(self.nusc.get('ego_pose', sd_record['ego_pose_token']))
                
                # Transform chain: Sweep Sensor -> Sweep Ego -> Global -> Ref Ego -> Ref Sensor.
                # a. Transform to ego frame at sweep time.
                pts = (curr_cs_r @ xyz.T + curr_cs_t) 
                # b. Transform to global frame.
                pts = (curr_pose_r @ pts + curr_pose_t)
                # c. Transform to current ego frame (using R^T = R^-1).
                pts = ref_pose_r.T @ (pts - ref_pose_t)
                # d. Transform to current lidar frame.
                pts = ref_cs_r.T @ (pts - ref_cs_t)
                
                points_fused = pts.T # [N, 3]

            # Build 5D points: [X, Y, Z, Intensity, Time_Lag].
            lags = time_lag.repeat(points_fused.size(0), 1)
            pc_5d = torch.cat([points_fused, intensity, lags], dim=1)
            all_points_5d.append(pc_5d)
            last_pc_5d = pc_5d

            # Move pointer to previous sweep.
            curr_sd_token = sd_record['prev']

        
        actual_num = len(all_points_5d)
        required_num = sweeps_num + 1
        if actual_num < required_num:
            # Compute how many sweeps need to be padded.
            num_to_pad = required_num - actual_num
            for _ in range(num_to_pad):
                # Pad using the last valid point cloud.
                all_points_5d.append(last_pc_5d.clone())

        # 3. Merge all point clouds.
        fused_pc = torch.cat(all_points_5d, dim=0)
        
        # 4. Sampling and padding.
        total_num_points = fused_pc.size(0)
        if total_num_points < target_num_points:
            padding = (0, 0, 0, target_num_points - total_num_points)
            fused_pc = F.pad(fused_pc, padding, value=0)
        else:
            # Random sampling.
            indices = torch.randperm(total_num_points)[:target_num_points]
            fused_pc = fused_pc[indices]

        # Return fused point cloud [max_N, 5].
        return fused_pc
    
    def get_transform_w2ego(self, sample, inverse = False):
        if 'data' in sample:
            lidar_token = sample['data']['LIDAR_TOP']
        else:
            token = sample['token'][-1] if isinstance(sample['token'], list) else sample['token']
            full_sample = self.nusc.get('sample', token)
            lidar_token = full_sample['data']['LIDAR_TOP']
        sample_data = self.nusc.get('sample_data', lidar_token)
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        T_w2ego = transform_matrix(ego_pose['translation'], Quaternion(ego_pose["rotation"]), inverse = inverse)
        # inverse=False: world -> ego
        # inverse=True: ego -> world
        return T_w2ego
    
    def get_waypoints(self, sample, T_ego2w_cur):
        T_w2ego = self.get_transform_w2ego(sample)
        T_cur2ego = T_ego2w_cur @ T_w2ego # currnet frame as origin
        theta = quaternion_yaw(Quaternion(matrix = T_cur2ego)) # yaw angle in radians
        
        origin = np.array(T_cur2ego[:3, 3])
        waypoint = torch.tensor([origin[0], origin[1], theta]).unsqueeze(0) # [1, waypoint]
        return waypoint

    def get_ego_state(self, sample):
        timestamp = self.nusc.get('ego_pose', sample['data']['LIDAR_TOP'])['timestamp']

        scene_token = sample['scene_token']
        scene_name = [s['name'] for s in self.nusc.scene if s['token'] == scene_token][0]

        pose_msgs = self.can_bus.get_messages(scene_name, 'pose')
        closest = min(pose_msgs, key=lambda x: abs(x['utime'] - timestamp))
        velocity_vector = torch.tensor(closest['vel'][:2], dtype=torch.float32)
        accel = torch.tensor(closest['accel'][:2], dtype=torch.float32)
        yaw_rate = torch.tensor(closest['rotation_rate'][2], dtype=torch.float32)

        velocity = torch.norm(velocity_vector)

        return velocity, accel, yaw_rate

    def get_instance(self, sample):
        token = sample['token'][-1] if isinstance(sample['token'], list) else sample['token']
        sample_record = self.nusc.get('sample', token)
        # lidar_token = sample['data']['LIDAR_TOP']
        lidar_token = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', lidar_token['ego_pose_token'])
        all_instances = []

        for ann_token in sample['anns']:
            ann_record = self.nusc.get('sample_annotation', ann_token)
            visible = int(ann_record['visibility_token'])
            if visible < 2: # 0: unknown, 1: not_visible, 2: partly, 3: fully
                continue

            box = Box(ann_record['translation'], ann_record['size'], Quaternion(ann_record['rotation']))

            # Step 1: global -> ego
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)

            # Filter out instances that are too far away
            distance = np.linalg.norm(box.center[:2])
            if distance > 30:
                continue

            raw_cat = ann_record['category_name']
            parts = raw_cat.split('.')
            # Use parts[1] if available, otherwise fallback to the first part
            category = parts[1] if len(parts) > 1 else parts[0]

            yaw = box.orientation.yaw_pitch_roll[0]
            # bbox format: [x, y, z, w, l, h, yaw]
            bbox_inf = [
                round(float(box.center[0]), 2), round(float(box.center[1]), 2), round(float(box.center[2]), 2),
                round(float(box.wlh[0]), 2), round(float(box.wlh[1]), 2), round(float(box.wlh[2]), 2),
                round(float(yaw), 2)
            ]
            all_instances.append({
                # 'sample_token': token,
                # 'instance_token': ann_record['instance_token'],
                'label': category,
                'bbox': bbox_inf,
                'distance': distance
            })

        # Sort instances by distance for better readability in the JSONL
        all_instances = sorted(all_instances, key=lambda x: x['distance'])

        return all_instances

    def classify_command(self, future_waypoints, x_stop_th=5.0, y_slight_th=2.0, y_hard_th=5.0):
        # future_waypoints shape is [future_frames, 3], where 3 is (x, y, yaw)
        data = future_waypoints.cpu().numpy() if isinstance(future_waypoints, torch.Tensor) else future_waypoints
        # Get the very last point from the tensor/array
        x_last = data[-1, 0]
        y_last = data[-1, 1]

        if x_last <= x_stop_th:
            return "<Decelerate_Stop>"
        
        # Classification based on lateral (y) displacement
        if y_last > y_hard_th:
            return "<Hard_Left>"
        elif y_slight_th < y_last <= y_hard_th:
            return "<Slight_Left>"
        elif -y_slight_th <= y_last <= y_slight_th:
            return "<Keep_Straight>"
        elif -y_hard_th <= y_last < -y_slight_th:
            return "<Slight_Right>"
        else: 
            return "<Hard_Right>"

def farthest_point_sampling(points, npoint):
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()

    device = points.device
    N, C = points.shape

    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)[0]

    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest].view(1, C)  # [1, C]
        dist = torch.sum((points - centroid) ** 2, dim=1)  # [N]
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=0)[1]

    sampled_points = points[centroids]  # [npoint, C]
    return sampled_points
