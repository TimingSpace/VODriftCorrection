'''
road segmentation
'''

import numpy as np
class RoadSegmentor:
    # initial road normal and height,max_width
    def __init__(self,road_normal,road_height,road_max_width,intrinsic,car=[1.7,4,5]):
        self.road_normal = road_normal
        self.road_height = road_height
        self.road_max_width = road_max_width
        self.road_color_model = np.ones(256)/256.0
        self.unroad_color_model=np.ones(256)/256.0
        self.road_confidence        = 0

        self.intrinsic = intrinsic
        self.car = car
        self.road_init_mask = []
        self.unroad_init_mask = []
        self.road_prab = []
    def update(self,img,road_mask,unroad_mask):
        road_color_model = self.distribution(img,road_mask)
        unroad_color_model = self.distribution(img,unroad_mask)
        conf = self.confidence(road_color_model,unroad_color_model)
        print(conf)
        self.road_color_model = (self.road_confidence*self.road_color_model+conf*road_color_model)/(self.road_confidence+conf)
        self.unroad_color_model =(self.road_confidence*self.unroad_color_model+conf*unroad_color_model)/(self.road_confidence+conf)
        #self.road_confidence = max(self.road_confidence,conf)+0.2*min(self.road_confidence,conf)

    def detect(self,img,prior=0.5):
        road_prab = np.zeros((img.shape[0],img.shape[1]))
        for i_row in range(img.shape[0]):
            for i_col in range(img.shape[1]):
                color = img[i_row,i_col]
                prior_ = prior
                if type(prior) is not float:
                    prior_= prior[i_row,i_col]
                p_xr = prior_
                p_xnr= 1 -prior_
                for i_color in range(0,len(color)):
                    p_xr *= self.road_color_model[i_color][color[i_color]]
                    p_xnr*= self.unroad_color_model[i_color][color[i_color]]
                road_prab[i_row,i_col] = p_xr/(p_xr+p_xnr+0.000000000001)
        self.road_prab = road_prab
        return road_prab
    # mean kl divergence
    def confidence(self,model_1,model_2,method='corr'):
        import scipy.special as ss
        conf = 0
        count= 0
        for a,b in zip(model_1,model_2):
            if method == 'corr':
                corr =  np.corrcoef(a,b)
                conf += 0.5-0.5*corr[0,1]
            elif method == 'kl_div':
                conf += np.mean(ss.kl_div(a,b))
            count+= 1
        return conf/count

    def distribution(self,img,mask):
        if len(img.shape)==2:
            img_l = img.reshape(-1)[mask.reshape(-1)==1]
            hist,x_axis = np.histogram(img_l,bins=range(-1,256),density=True)
            return np.array([hist])
        else:
            img_b = img[:,:,0].reshape(-1)[mask.reshape(-1)==1]
            hist_b,x_axis = np.histogram(img_b,bins=range(-1,256),density=True)
            img_g = img[:,:,1].reshape(-1)[mask.reshape(-1)==1]
            hist_g,x_axis = np.histogram(img_g,bins=range(-1,256),density=True)
            img_r = img[:,:,2].reshape(-1)[mask.reshape(-1)==1]
            hist_r,x_axis = np.histogram(img_r,bins=range(-1,256),density=True)
            return np.array([hist_b,hist_g,hist_r])

    def process(self,img,path=None):
        if path is not None:
            road_mask = self.project_prab(img,path,self.intrinsic,self.car)
            self.road_init_mask = road_mask
            unroad_mask = np.zeros(img.shape[:2])
            unroad_mask[0:unroad_mask.shape[0]//2,:] =1
            self.unroad_init_mask = unroad_mask
            self.update(img,road_mask,unroad_mask)
            return self.detect(img,road_mask)
        else:
            return self.detect(img)
    def EMProcess(self,img,path=None):
        if path is not None:
            road_mask = self.project_prab(img,path,self.intrinsic,self.car)
            self.road_init_mask = road_mask
            unroad_mask = np.zeros(img.shape[:2])
            unroad_mask[0:unroad_mask.shape[0]//2,:] =1
            self.unroad_init_mask = unroad_mask
            for i in range(3):
                self.update(img,road_mask,unroad_mask)
                road_mask = self.detect(img,road_mask)
                road_mask,res = self.estimate_road(road_mask)
                self.visualize(img.copy())
                road_mask[road_mask>0.5] = 1
                unroad_mask[road_mask<0.5] = 1
                self.road_init_mask = road_mask
                self.unroad_init_mask = unroad_mask
        else:
            return self.detect(img)

# img reference image [w,h,3]
# path                [n,3]
# intrinsic           [3,3]
# car [height,width,length]  [3]
    def project(self,img,path_,intrinsic,car):
        path = path_.copy()
        path[:,1] +=car[0]
        uvz  = intrinsic@path.transpose(1,0)
        u    = uvz[0,:]/uvz[2,:]
        v    = uvz[1,:]/uvz[2,:]
        w    = (v - intrinsic[1,2])*intrinsic[0,0]*car[1]/(intrinsic[1,1]*car[0])
        h    = car[2]*car[0]*intrinsic[1,1]/(path[:,2]*(path[:,2]-car[2]))
        valid = (v<img.shape[0])&(v>0)&(path[:,2]-car[2]>0)
        path = path[valid,:]
        u = u[valid]
        v = v[valid]
        w = w[valid]
        h = h[valid]
        
        mask = np.zeros((img.shape[0],img.shape[1]))
        for i in range(0,len(u)):
            lt  = (int(u[i]-w[i]/2),int(v[i]))
            rl  = (int(u[i]+w[i]/2),int(v[i]+h[i]))
            mask[lt[1]:rl[1],lt[0]:rl[0]]=1
        return mask
    

    def project_prab(self,img,path_,intrinsic,car):
        path = path_.copy()
        path[:,1] +=car[0]
        uvz  = intrinsic@path.transpose(1,0)
        u    = uvz[0,:]/uvz[2,:]
        v    = uvz[1,:]/uvz[2,:]
        w    = (v - intrinsic[1,2])*intrinsic[0,0]*car[1]/(intrinsic[1,1]*car[0])
        h    = car[2]*car[0]*intrinsic[1,1]/(path[:,2]*(path[:,2]-car[2]))
        valid = (v<img.shape[0])&(v>0)&(path[:,2]-car[2]>0)
        path = path[valid,:]
        u = u[valid]
        v = v[valid]
        w = w[valid]
        h = h[valid]
        w_r    = w*self.road_max_width/car[1]
        w_diff = w_r - w
        mask = np.zeros((img.shape[0],img.shape[1]))
        for i in range(0,len(u)):
            lt  = (int(u[i]-w[i]/2),int(v[i]))
            rl  = (int(u[i]+w[i]/2),int(v[i]+h[i]))
            mask[lt[1]:rl[1],lt[0]:rl[0]]=1

            lt  = (int(u[i]-w_r[i]+w[i]/2),int(v[i]))
            rl  = (int(u[i]-w[i]/2),int(v[i]+h[i]))
            l  = max(0,lt[0])
            r  = min(img.shape[1],rl[0])
            up = max(0,lt[1])
            lo = min(img.shape[0],rl[1])
            if r<l or lo<up:
                continue
            shift_l = l - lt[0]
            shift_r = r - rl[0]
            shift_up=up - lt[1]
            shift_lo=lo - rl[1]
            mask_local = np.array([[pow(u/w_diff[i],2) for u in range(0,rl[0]-lt[0])]])
            mask_local = np.repeat(mask_local,rl[1]-lt[1],axis=0)
            mask[up:lo,l:r]=mask_local[shift_up:rl[1]-lt[1]+shift_lo,shift_l:rl[0]-lt[0]+shift_r]
            

            lt  = (int(u[i]+w[i]/2),int(v[i]))
            rl  = (int(u[i]+w_r[i]-w[i]/2),int(v[i]+h[i]))
            l  = max(0,lt[0])
            r  = min(img.shape[1],rl[0])
            up = max(0,lt[1])
            lo = min(img.shape[0],rl[1])
            if r<l or lo<up:
                continue
            shift_l = l - lt[0]
            shift_r = r - rl[0]
            shift_up=up - lt[1]
            shift_lo=lo - rl[1]
            mask_local = np.array([[pow(1-u/w_diff[i],2) for u in range(0,rl[0]-lt[0])]])
            mask_local = np.repeat(mask_local,rl[1]-lt[1],axis=0)
            mask[up:lo,l:r]=mask_local[shift_up:rl[1]-lt[1]+shift_lo,shift_l:rl[0]-lt[0]+shift_r]
                   
        return mask

    def estimate_road(self,prob,center=None,flag_vis =False):
        prob_new = np.zeros(prob.shape)
        res =[]
        import map_road as mr
        for i in range(prob.shape[0]//2,prob.shape[0]):
            a,b,p = mr.estimate_road(prob[i,:])
            res.append([a,b,p,i])
            prob_new[i,a:b]=p
        return prob_new,res


    def visualize(self,img_bgr):
        import matplotlib.pyplot as plt
        ax1 = plt.subplot(511)
        img_bgr[:,:,1] = 255*self.road_init_mask
        ax1.imshow(img_bgr[:,:,::-1])
        ax1.set_title('drive path')
        ax1.margins(100)
        ax1 = plt.subplot(512)
        img_bgr[:,:,1] = 255*self.unroad_init_mask
        ax1.imshow(img_bgr[:,:,::-1])
        ax1.set_title('road probability')
        
        ax1 = plt.subplot(513)
        img_bgr[:,:,1] = 255*self.road_prab
        #ax1.imshow(img_bgr[:,:,::-1])
        ax1.imshow(self.road_prab)
        ax1 = plt.subplot(514)  
        ax1.plot(self.road_color_model[0],'b')
        ax1.plot(self.road_color_model[1],'g')
        ax1.plot(self.road_color_model[2],'r')
        ax1 = plt.subplot(515)
        ax1.set_ylim(0,0.02)
        ax1.plot(self.unroad_color_model[0],'b')
        ax1.plot(self.unroad_color_model[1],'g')
        ax1.plot(self.unroad_color_model[2],'r')
        plt.show()

def main():
# load image
    import sys
    import cv2
    images_path = sys.argv[1]
    images      = open(images_path)
    image_name  = images.readline() # first line is not pointing to a image, so we read and skip it
    image_names = images.read().split('\n')
    
    poses_m       = np.loadtxt(sys.argv[2])
    image_id = 0
    begin_id = 0
    plane = [0,1,0,1.7]
    intrinsic = np.array([[719,0,607],[0,719,185],[0,0,1]])


# road segmentation
    road_segmentor = RoadSegmentor(plane[0:3],plane[3],10,intrinsic)
    for image_name in image_names:
        if image_id<begin_id:
            image_id+=1
            continue
        print(image_name)
        if len(image_name) == 0:
            break
        img = cv2.imread(image_name,0)
        img_bgr = cv2.imread(image_name)
        img_bgr_curr = cv2.imread(image_names[image_id+50])
        road_segmentor.EMProcess(img_bgr,poses_m[0:100,3:12:4])
        road_segmentor.EMProcess(img_bgr_curr)
        road_segmentor.visualize(img_bgr)




if __name__ == '__main__':
    main()
