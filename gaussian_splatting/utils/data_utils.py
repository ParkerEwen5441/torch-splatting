import os
import json
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from einops import rearrange

# change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
def read_camera(folder):
    """
    read camera from json file
    """
    scene_info = json.load(open(os.path.join(folder, 'info.json')))
    # scene_info = json.load(open(os.path.join(folder, 'test.json')))
    try:
        intrinsics = scene_info['intrinsics']
    except:
        pass

    rgb_files = []
    poses = []
    max_depths = []
    for item in scene_info['images']:
        rgb_files.append(os.path.join(folder, item['rgb']))

        # Habitat coords to camera coords
        h2c = np.eye(4)
        h2c[1,1] = -1
        h2c[2,2] = -1 

        # World to agent transform
        w2a = np.array(item['pose'])

        # Agent to world transform
        a2c = np.eye(4)
        a2c[:3, 3] = np.array([0, 1.5, 0])

        # Camera to world
        c2w = w2a @ a2c @ h2c

        poses.append(c2w)
        max_depths.append(np.array(item['max_depth']))
    return rgb_files, poses, intrinsics, max_depths

def read_all(folder, resize_factor=1.):
    """
    read source images from a folder
    """
    src_rgb_files, src_poses, intrinsics, max_depths = read_camera(folder)

    src_cameras = []
    src_rgbs = []
    src_alphas = []
    src_depths = []
    src_semantics = []


    for src_rgb_file, src_pose, max_depth in zip(src_rgb_files, src_poses, max_depths):
        src_rgb , src_depth, src_alpha, src_semantic, src_camera = \
        read_image(src_rgb_file, 
                   src_pose, 
                   max_depth=max_depth,
                   intrinsics=intrinsics,
                   resize_factor=resize_factor)

        src_rgbs.append(src_rgb)
        src_depths.append(src_depth)
        src_alphas.append(src_alpha)
        src_cameras.append(src_camera)
        src_semantics.append(src_semantic)
    
    src_alphas = torch.stack(src_alphas, axis=0)
    src_depths = torch.stack(src_depths, axis=0)
    src_rgbs = torch.stack(src_rgbs, axis=0)
    src_semantics = torch.stack(src_semantics, axis=0)
    src_cameras = torch.stack(src_cameras, axis=0)

    src_rgbs = src_alphas[..., None] * src_rgbs + (1-src_alphas)[..., None]

    return {
        "rgb": src_rgbs[..., :3],
        "camera": src_cameras,
        "depth": src_depths,
        "alpha": src_alphas,
        "semantic": src_semantics
    }


def read_image(rgb_file, pose, max_depth, intrinsics, resize_factor=1., white_bkgd=True):
    rgb = torch.from_numpy(imageio.imread(rgb_file).astype(np.float32) / 255.0)
    depth = torch.from_numpy(imageio.imread(rgb_file[:-7]+'depth.png').astype(np.float32) / 255.0 * max_depth)
    alpha = torch.from_numpy(imageio.imread(rgb_file[:-7]+'alpha.png').astype(np.float32) / 255.0)
    semantic = torch.from_numpy(imageio.imread(rgb_file[:-7]+'semantic.png').astype(np.float32) / 255.0)
    # semantic = torch.from_numpy(imageio.imread(rgb_file[:-7]+'alpha.png').astype(np.float32) / 255.0)

    image_size = rgb.shape[:2]
    intrinsic = np.eye(4)
    intrinsic[:3,:3] = np.array(intrinsics)

    if resize_factor != 1:
        image_size = image_size[0] * resize_factor, image_size[1] * resize_factor 
        intrinsic[:2,:3] *= resize_factor
        resize_fn = lambda img, resize_factor: F.interpolate(
                img.permute(0, 3, 1, 2), scale_factor=resize_factor, mode='bilinear',
            ).permute(0, 2, 3, 1)
        
        rgb = rearrange(resize_fn(rearrange(rgb, 'h w c -> 1 h w c'), resize_factor), '1 h w c -> h w c')
        depth = rearrange(resize_fn(rearrange(depth, 'h w -> 1 h w 1'), resize_factor), '1 h w 1 -> h w')
        alpha = rearrange(resize_fn(rearrange(alpha, 'h w -> 1 h w 1'), resize_factor), '1 h w 1 -> h w')
        semantic = rearrange(resize_fn(rearrange(semantic, 'h w -> 1 h w 1'), resize_factor), '1 h w 1 -> h w')

    camera = torch.from_numpy(np.concatenate(
        (list(image_size), intrinsic.flatten(), pose.flatten())
    ).astype(np.float32))
    
    if white_bkgd:
        rgb = alpha[..., None] * rgb + (1-alpha)[..., None]

    return rgb, depth, alpha, semantic, camera