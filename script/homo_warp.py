# this code is borrowed from `https://github.com/ClementPinard/SfmLearner-Pytorch`
from __future__ import division
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

pixel_coords = None


# generate the pixel coords pixel_coords[i,j] = [i,j] with the same shape of input
def set_id_grid(img):
    global pixel_coords
    b,_,h, w = img.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).float()  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).float()  # [1, H, W]
    ones = torch.ones(1,h,w).float()

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def homo_project(pixel_coords,homo_mat):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        pixel_coords: pixel coordinates  -- [B, 3, H, W]
        homo_mat: [B,3,3]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = pixel_coords.size()
    pixel_coords_flat = pixel_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    pcoords = homo_mat@pixel_coords_flat
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords_n = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords_n.reshape(b,h,w,2)

def so2mat(angle):
    """Convert so3 angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis  -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the so3 angles -- size = [B, 3, 3]
    """
    return torch.Tensor(R.from_rotvec(angle).as_dcm())


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def homo_vec2mat(vec,plane,k1,k2,rotation_mode='so3'):
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    normal =plane[:,:3].unsqueeze(1)
    plane_h = plane[:,3]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'so3':
        rot_mat = so2mat(rot)
    rot_mat_inverse = rot_mat.inverse()
    homo_mat = rot_mat_inverse - rot_mat_inverse@translation@normal/plane_h
    return k1@homo_mat.inverse()@k2.inverse()


def homograph_warp(img,pose,plane,intrinsics_a,intrinsics_b,rotation_mode='so3',padding_mode='zeros'):
    """
    homography warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        pose: 6DoF pose parameters from target to source(target pose relative to source frame) -- [B, 6]
        plane: plane function nX-h=0 [B,4]
        intrinsics_a: source camera intrinsic matrix -- [B, 3, 3]
        intrinsics_b: target camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """

    check_sizes(img, 'img', 'B3HW')
    check_sizes(plane, 'depth', 'B4')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics_a, 'intrinsics_a', 'B33')
    check_sizes(intrinsics_b, 'intrinsics_b', 'B33')

    batch_size, _, img_height, img_width = img.size()

    b, h, w = batch_size,img_height,img_width
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(img)

    homo_mat = homo_vec2mat(pose,plane,intrinsics_a,intrinsics_b,rotation_mode)
    print(homo_mat.squeeze())
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w)  # [B, 3, H,W]
    src_pixel_coords = homo_project(current_pixel_coords,homo_mat)  # [B,H,W,2]
    projected_img = None
    if torch.__version__ !='1.1.0.post2':
        projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode,align_corners=False)

    else:
        projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)
    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points


if __name__ == '__main__':
    import sys
    import cv2
    import numpy as np
    img = cv2.imread(sys.argv[1])
    img_t = torch.Tensor(img.transpose(2,0,1)).unsqueeze(0)
    pose = torch.Tensor([0,0,35,-np.pi/2,0,0]).unsqueeze(0) # first t than rotation
    plane = torch.Tensor([0,1,0,1.7]).unsqueeze(0)
    intrinsic = torch.Tensor([[719,0,607],[0,719,185],[0,0,1]]).unsqueeze(0)
    intrinsic_2 = torch.Tensor([[10,0,607],[0,10,185],[0,0,1]]).unsqueeze(0)
    img_homo,valid = homograph_warp(img_t,pose,plane,intrinsic,intrinsic_2)
    img_homo = np.array(img_homo.squeeze()).transpose(1,2,0)
    cv2.imshow('homo',img_homo/255)
    cv2.waitKey()
