import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os, random
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from .ray_utils import *


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)


class LLFFDataset(Dataset):
    '''
    The dataset structure:
    root_dir
        - images
        - segmentations
            - classes.txt
            - 00
                - [class 1 name].png (the segmentation mask of class 1)
                - [class 2 name].png (the segmentation mask of class 2)
                ...
            - 03
            ...
        - sparse
        - poses_bounds.npy
        - colmap_output.txt
        - poses_bounds.npy
    '''
    def __init__(self, datadir, patch_size=256, patch_stride=1, split='train', downsample=8, is_stack=False, clip_input=1.):

        self.clip_input = clip_input
        self.root_dir = datadir
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.white_bg = False

        # self.near_far = [0.0, 1.0]
        # self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        self.near_far = [-0.5, 1.5]
        self.scene_bbox = torch.tensor([[-2, -2, -1.0], [2, 2, 1.0]])

        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
        # load full resolution image then resize
        if self.split in ['train', 'test']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)
        hwf = poses[:, :, -1]

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor

        # build rendering path
        N_views, N_rots = 120, 2
        tt = self.poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        up = normalize(self.poses[:, :3, 1].sum(0))
        rads = np.percentile(np.abs(tt), 90, 0)

        self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views)
        # self.render_path = get_interpolation_path(self.poses)

        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = self.img_wh
        self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)

        average_pose = average_poses(self.poses)
        dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)
        # i_test = np.arange(0, self.poses.shape[0], self.hold_every)  # [np.argmin(dists)]
        # self.img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))
        self.img_list = np.arange(len(self.poses)) if self.split == 'train' else [0]

        if self.clip_input != 1.:
            # Calculate the number of indices to select
            num_indices = int(self.clip_input * len(self.img_list))
            # Randomly select the indices
            selected_indices = random.sample(range(len(self.img_list)), num_indices)
            # Retrieve the corresponding elements
            self.img_list = [self.img_list[i] for i in selected_indices]

        # use first N_images-1 to train, the LAST is val
        self.all_rays = []
        self.all_rgbs = []
        for i in self.img_list:
            image_path = self.image_paths[i]
            c2w = torch.FloatTensor(self.poses[i])

            img = Image.open(image_path).convert('RGB')
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)

            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            self.all_rgbs += [img]
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
            # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        all_rays = self.all_rays
        all_rgbs = self.all_rgbs

        self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w,6)
        self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)

        if self.is_stack:
            self.all_rays_stack = torch.stack(all_rays, 0).reshape(-1,*self.img_wh[::-1], 6)  # (len(self.meta['frames]),h,w,6)
            self.all_rgbs_stack = torch.stack(all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)

    def read_classes_names(self):
        # read class names
        with open(os.path.join(self.root_dir, 'segmentations/classes.txt'), 'r') as f:
            lines = f.readlines()
            self.classes = [line.strip() for line in lines]
            self.classes.sort()
    
    def read_segmentation_maps(self):
        segmentation_path = os.path.join(self.root_dir, 'segmentations')
        # get a list of all the folders in the directory
        folders = [f for f in os.listdir(segmentation_path) if os.path.isdir(os.path.join(segmentation_path, f))]

        seg_maps = []
        self.idxes = [] # the idx of the test imgs
        for folder in folders:
            self.idxes.append(int(folder))  # to get the camera id
            seg_for_one_image = []
            for class_name in self.classes:
                # check if the seg map exists
                seg_path = os.path.join(self.root_dir, f'segmentations/{folder}/{class_name}.png')
                if not os.path.exists(seg_path):
                    raise Exception(f'Image {class_name}.png does not exist')
                img = Image.open(seg_path).convert('L')
                # resize the seg map
                if self.downsample != 1.0:
                    img = img.resize(self.img_wh, Image.NEAREST) # [W, H]
                img = (np.array(img) / 255.0).astype(np.int8) # [H, W]
                img = img.flatten() # [H*W]
                seg_for_one_image.append(img)

            seg_for_one_image = np.stack(seg_for_one_image, axis=0)
            seg_for_one_image = seg_for_one_image.transpose(1, 0)
            seg_maps.append(seg_for_one_image)

        self.seg_maps = np.stack(seg_maps, axis=0) # [n_frame, H*W, n_class]

    @torch.no_grad()
    def read_clip_features_and_relevancy_maps(self, feature_dir, text_features, test_prompt=None):
        '''
        the input text_features are already normalized
        '''

        feature_paths = sorted(glob.glob(f'{feature_dir}/*'))
        features, relevancy_maps = [], []
        
        print('Reading CLIP feaures...')
        for i in tqdm(self.img_list):
            if test_prompt is not None and i != int(test_prompt): continue

            feature = torch.load(feature_paths[i], map_location='cuda') # [scale(=3), dim, H, W]
            if feature.size(-2) != self.img_wh[1] or feature.size(-1) != self.img_wh[0]:
                feature = TF.resize(feature, size=(self.img_wh[1], self.img_wh[0]))
            feature = F.normalize(feature, dim=1) 
            features.append(feature.cpu())
            # compute relevancy map
            relevancy_map = torch.einsum('cd,sdhw->schw', text_features, feature) # [scale(=3), num_classes, H, W]
            relevancy_map_reshaped = relevancy_map.reshape(relevancy_map.size(0), relevancy_map.size(1), -1)
            relevancy_map_min = torch.min(relevancy_map_reshaped, dim=-1, keepdim=True).values[..., None]
            relevancy_map_max = torch.max(relevancy_map_reshaped, dim=-1, keepdim=True).values[..., None]
            relevancy_map = (relevancy_map - relevancy_map_min) / (relevancy_map_max - relevancy_map_min)
            relevancy_maps.append(relevancy_map.cpu())

            # save the relevancy map as image
            if test_prompt is not None:
                os.makedirs('clip_features/clip_relevancy_maps', exist_ok=True)
                for i in range(relevancy_map.size(0)):
                    for j in range(relevancy_map.size(1)):
                        img = relevancy_map[i,j].cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                        img = Image.fromarray(img)
                        img.save(f'clip_features/clip_relevancy_maps/{i}_{j}.png')
                exit()

        features = torch.stack(features, dim=0) # [n_frame, scale(=3), dim, H, W]
        relevancy_maps = torch.stack(relevancy_maps, dim=0) # [n_frame, scale(=3), num_classes, H, W]

        features = features.permute(0,3,4,1,2) # [n_frame, H, W, scale(=3), dim]
        self.all_features = torch.reshape(features, (-1, features.size(-2), features.size(-1))) # [-1, scale, dim]
        relevancy_maps = relevancy_maps.permute(0,3,4,1,2) # [n_frame, H, W, scale(=3), num_classes]
        self.all_relevancies = torch.reshape(relevancy_maps, (-1, relevancy_maps.size(-2), relevancy_maps.size(-1))) # [-1, scale, num_classes]


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rays_stack)

    def __getitem__(self, i):
        '''
        Only used for patch sampling
        '''

        h_idx = random.randint(0, self.img_wh[1]-self.patch_size)
        w_idx = random.randint(0, self.img_wh[0]-self.patch_size)
        ray_sample = self.all_rays_stack[i]
        rays = ray_sample[h_idx:h_idx+self.patch_size:self.patch_stride, w_idx:w_idx+self.patch_size:self.patch_stride, :]
        # down size rays to match dino feature downsize
        avg_pool = torch.nn.AvgPool2d(8, ceil_mode=True)
        rays = avg_pool(rays.permute(2,0,1)[None,...]).squeeze(0).permute(1,2,0)
        
        rgb_sample = self.all_rgbs_stack[i]
        rgbs = rgb_sample[h_idx:h_idx+self.patch_size:self.patch_stride, w_idx:w_idx+self.patch_size:self.patch_stride, :]
        
        
        return {
            'rays': rays, # [patch_size//8, patch_size//8, 6]
            'rgbs': rgbs  # [pathc_size, patch_size, 3]
        }
