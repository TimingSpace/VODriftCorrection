'''
real time road segmentation
@author Xiangwei Wang wangxiangwei.cpp@gmail.com
'''
import sys
import numpy as np
import cv2
import param
from road_segmentation import RoadSegmentor
def main():
    # load image
    images_path = sys.argv[1]
    images      = open(images_path)
    image_name  = images.readline() # first line is not pointing to a image, so we read and skip it
    image_names = images.read().split('\n')
    
    poses_m       = np.loadtxt(sys.argv[2])
    image_id = 0
    begin_id = 0
    plane = [0,1,0,1.7]
    intrinsic = np.array([[719,0,607],[0,719,185],[0,0,1]])
    road_segmentor = RoadSegmentor(plane[0:3],plane[3],10,intrinsic)
    for image_name in image_names:
        if image_id<begin_id:
            image_id+=1
            continue
        print(image_name)
        if len(image_name) == 0:
            break
        img_bgr = cv2.imread(image_name)
        img_bgr_cur = cv2.imread(image_names[image_id+5])
        poses_restart = restart_path(poses_m[image_id+1:image_id+100,:])

        road_segmentor.EMProcess(img_bgr,poses_restart[:,3:12:4])
        road_segmentor.EMProcess(img_bgr_cur)
        image_id+=1
        


def l2m(pose):
    m = np.eye(4)
    m[0:3,:] = pose.reshape(3,4)
    return np.matrix(m)
def m2l(pose):
    return np.array(pose[:3,:]).reshape(1,-1)[0]

def restart_path(poses):
    start_inv = l2m(poses[0,:]).I
    new_poses = poses.copy()
    for i in range(1,poses.shape[0]):
        curr = l2m(poses[i,:])
        res  = start_inv*curr
        new_poses[i,:] = m2l(res)
    return new_poses[1:,:]

if __name__ == '__main__':
    main()


