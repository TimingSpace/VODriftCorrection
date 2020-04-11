'''
real time road segmentation
@author Xiangwei Wang wangxiangwei.cpp@gmail.com
'''
import sys
import numpy as np
import cv2
import param
import transformation as tf
import homo_warp
import pytorch_ssim
import torch



def optimize(img_1,img_2,motion,plane,intrinsic,mask=None):
    img_1 = itT(img_1)/255
    img_2 = itT(img_2)/255
    motion = tT(motion)
    motion.requires_grad = True
    plane  = tT(plane)
    plane.requires_grad  = True
    intrinsic = tT(intrinsic)
    if mask is not None:
        mask = tT(mask)
    mask.requires_grad = True
    ssim_loss = pytorch_ssim.SSIM()
    optimizer =  torch.optim.Adam([motion,plane],lr=0.00001)
    epoch = 0
    while epoch<100:
        img_homo,valid = homo_warp.homograph_warp(img_1,motion,plane,intrinsic,intrinsic,rotation_mode='euler')
        if mask is not None:
            ssim_out = -ssim_loss(img_2,img_homo,mask)
        ssim_value = - ssim_out.data.item()
        print(ssim_value)
        ssim_out.backward()
        optimizer.step()
        print(motion,plane)
        ssim_map = pytorch_ssim.ssim_map(img_2,img_homo,11,mask)
        img_show = img_homo.detach().numpy().squeeze().transpose(1,2,0)
        img_ssim = ssim_map.detach().numpy().squeeze().transpose(1,2,0)
        epoch +=1
        cv2.imshow('img',img_show)
        cv2.imshow('ssim',img_ssim)
    
        key = cv2.waitKey(1)
        if key&255==ord('q'):
            break

def main():
    # load image
    images_path = sys.argv[1]
    images      = open(images_path)
    image_name  = images.readline() # first line is not pointing to a image, so we read and skip it
    image_names = images.read().split('\n')
    
    poses_m       = np.loadtxt(sys.argv[2])
    motions_m     = tf.pose2motion(poses_m)
    motions_se    = tf.SEs2ses(motions_m)
    image_id = 0
    begin_id = 0
    plane = [0,1,0,1.7]
    intrinsic = np.array([[719,0,607],[0,719,185],[0,0,1]])
    for image_name in image_names:
        if image_id<begin_id:
            image_id+=1
            continue
        print(image_name)
        if len(image_name) == 0:
            break
        img = cv2.imread(image_name,0)
        #img = cv2.resize(img,(param.img_w,param.img_h))
        img_bgr = cv2.imread(image_name)
        img_bgr_next = cv2.imread(image_names[image_id+1]) 

        poses_restart = restart_path(poses_m[image_id+1:image_id+50,:])
        mask = project(img_bgr,poses_restart[:,3:12:4],intrinsic,[1.7,4,5])

        mask_unroad = np.zeros(img.shape)
        mask_unroad[0:mask_unroad.shape[0]//2,:] =1
        road_segmentation(img,mask,mask_unroad)
        #road_segmentation(img_bgr,mask,mask_unroad)
        #optimize(img_bgr,img_bgr_next,motions_se[image_id],plane,intrinsic,mask)
        '''
        img_homo,valid =homo_warp.homograph_warp(tT(img_bgr.transpose(2,0,1)),tT(motions_se[image_id]),tT(plane),tT(intrinsic),tT(intrinsic))
        img_homo = np.array(img_homo.squeeze()).transpose(1,2,0)
        img_diff = np.abs(img_homo - img_bgr_next)
        img_ssim = pytorch_ssim.ssim_map(itT(img_homo)/255,itT(img_bgr_next)/255)
        cv2.imshow('homo',img_homo/255)
        #img_bgr = cv2.resize(img_bgr,(param.img_w,param.img_h))
        cv2.imshow('image',img_bgr)
        cv2.imshow('image_next',img_bgr_next)
        cv2.imshow('diff',img_diff/255)
        cv2.imshow('ssim',np.mean(tI(img_ssim),2))
        '''
        
        image_id+=1

def itT(a):
    return torch.Tensor(a.transpose(2,0,1)).unsqueeze(0)
def tT(a):
    return torch.Tensor(a).unsqueeze(0)
def tI(a):
    return a.detach().numpy().squeeze().transpose(1,2,0)




def road_segmentation(img,mask_road,mask_unroad,prior=0.5):
    dis_road = distribution(img,mask_road)
    dis_unroad = distribution(img,mask_unroad)
    print(len(img.shape))
    road_prab=[]
    if len(img.shape)==2:
        dis_road_color = prior*dis_road[0]/(prior*dis_road[0]+(1-prior)*dis_unroad[0])
        print(dis_road_color.shape)
        road_prab  = np.array([dis_road_color[d] for d in img.reshape(-1)]).reshape(img.shape)
    elif len(img.shape)==3:
        road_prab = np.zeros((img.shape[0],img.shape[1]))
        dis_road_color = prior*dis_road[0]/(prior*dis_road[0]+(1-prior)*dis_unroad[0])
        for i_row in range(img.shape[0]):
            for i_col in range(img.shape[1]):
                color = img[i_row,i_col]
                p_xr = prior*dis_road[0][color[0]]*dis_road[1][color[1]]*dis_road[2][color[2]]
                p_xnr= (1-prior)*dis_unroad[0][color[0]]*dis_unroad[1][color[1]]*dis_unroad[2][color[2]]
                road_prab[i_row,i_col] = p_xr/(p_xr+p_xnr)
        print(road_prab.shape)
    print(road_prab.shape)
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(311)
    pos1 = ax1.imshow(road_prab,cmap='gray')
    ax1.set_title('Road Probability')
    ax1.margins(0)
    ax2 = plt.subplot(323)
    ax2.plot(dis_road[0],'g',label='P(c|r)')
    ax2.plot(dis_unroad[0],'r',label='P(c|~r)')
    ax2.set_ylim(0,0.05)
    ax2.legend()
    ax3 = plt.subplot(324)
    ax3.plot(dis_road_color,'b',label='P(r|c)')
    ax3.legend()
    ax4 = plt.subplot(337)
    ax4.set_title('input image')
    ax4.imshow(img)
    ax5 = plt.subplot(338)
    ax5.set_title('Road mask')
    road_img = img.copy()
    road_img[mask_road==1]=255
    ax5.imshow(road_img)
    ax6 = plt.subplot(339)
    ax6.set_title('Non-Road mask')
    unroad_img = img.copy()
    unroad_img[mask_unroad==1]=0
    ax6.imshow(unroad_img)
    plt.show()


def distribution(img,mask):
    if len(img.shape)==2:
        img_l = img.reshape(-1)[mask.reshape(-1)==1]
        hist,x_axis = np.histogram(img_l,bins=range(-1,256),density=True)
        return [hist]
    else:
        img_b = img[:,:,0].reshape(-1)[mask.reshape(-1)==1]
        hist_b,x_axis = np.histogram(img_b,bins=range(-1,256),density=True)
        img_g = img[:,:,1].reshape(-1)[mask.reshape(-1)==1]
        hist_g,x_axis = np.histogram(img_g,bins=range(-1,256),density=True)
        img_r = img[:,:,2].reshape(-1)[mask.reshape(-1)==1]
        hist_r,x_axis = np.histogram(img_r,bins=range(-1,256),density=True)
        return [hist_b,hist_g,hist_r]

        


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

# img reference image [w,h,3]
# path                [n,3]
# intrinsic           [3,3]
# car [height,width,length]  [3]
def project(img,path,intrinsic,car):
    path[:,1] +=car[0]
    uvz  = intrinsic@path.transpose(1,0)
    u    = uvz[0,:]/uvz[2,:]
    v    = uvz[1,:]/uvz[2,:]
    w    = (v - intrinsic[1,2])*intrinsic[0,0]*car[1]/(intrinsic[1,1]*car[0])
    h    = car[2]*car[0]*intrinsic[1,1]/(path[:,2]*(path[:,2]-car[2]))
    valid = (v<img.shape[1])&(v>0)&(path[:,2]-car[2]>0)
    path = path[valid,:]
    u = u[valid]
    v = v[valid]
    w = w[valid]
    h = h[valid]
    print_data = np.ones((path.shape[0],7))
    print_data[:,:3]=path
    print_data[:,3] =u 
    print_data[:,4] =v 
    print_data[:,5] =w 
    print_data[:,6] =h 
    
    mask = np.zeros((img.shape[0],img.shape[1]))
    for i in range(0,len(u)):
        #cv2.circle(img,(int(u[i]),int(v[i])),3,(255,0,0),-1)
        lt  = (int(u[i]-w[i]/2),int(v[i]))
        rl  = (int(u[i]+w[i]/2),int(v[i]+h[i]))
        mask[lt[1]:rl[1],lt[0]:rl[0]]=1
        #print(lt,rl)
        #cv2.rectangle(img,lt,rl,(0,0,255))
    #cv2.imshow('img',mask)
    #cv2.waitKey(1)
    return mask


if __name__ == '__main__':
    main()
