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
        self.intrinsic_inverse = np.array(np.matrix(intrinsic).I)
        self.car = car
        self.road_init_mask = []
        self.unroad_init_mask = []
        self.road_prob = []
        self.road_prob_flat = []
        self.road_prob_ml = []

        self.road_xl =None
        self.road_xr =None

    def update(self,img,road_mask,unroad_mask):
        road_color_model = self.distribution(img,road_mask)
        unroad_color_model = self.distribution(img,unroad_mask)
        conf = self.confidence(road_color_model,unroad_color_model)
        print(conf)
        self.road_color_model = (self.road_confidence*self.road_color_model+conf*road_color_model)/(self.road_confidence+conf)
        self.unroad_color_model =(self.road_confidence*self.unroad_color_model+conf*unroad_color_model)/(self.road_confidence+conf)
        self.road_confidence = max(self.road_confidence,conf)+0.2*min(self.road_confidence,conf)

    def detect(self,img,prior=0.5):
        road_prob = np.zeros((img.shape[0],img.shape[1]))
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
                road_prob[i_row,i_col] = p_xr/(p_xr+p_xnr+0.000000000001)
        self.road_prob = road_prob
        return road_prob
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
            road_mask = self.project_prob(img,path,self.intrinsic,self.car)
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
            # get road prior probability
            road_mask=[]
            if self.road_xl is None:
                road_mask = self.project_prob(img,path,self.intrinsic,self.car)
            else:
                road_mask = self.project_prob_lr(img,path,self.intrinsic,self.car)

            self.road_init_mask = road_mask
            unroad_mask = np.zeros(img.shape[:2])
            unroad_mask[0:unroad_mask.shape[0]//2,:] =1
            unroad_mask[road_mask<0.3]=1
            self.unroad_init_mask = unroad_mask

            for i in range(1):
                # update road model 
                self.update(img,road_mask,unroad_mask)
                # detect road
                road_mask = self.detect(img,road_mask)
                #road_mask,res = self.estimate_road(road_mask)
                # calculate road parameter 
                self.road_calculation(road_mask,path)
                # update road mask 
                #road_mask[road_mask>0.5] = 1
                #unroad_mask[road_mask<0.3] = 1
                #self.road_init_mask = road_mask
                #self.unroad_init_mask = unroad_mask
            self.visualize(img)
        else:
            road_prab = self.detect(img)
            self.visualize(img)
            return road_prab

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
 
    def project_prob_lr(self,img,path_,intrinsic,car):
        path = path_.copy()
        path[:,1] +=car[0]
        uvz  = intrinsic@path.transpose(1,0)
        u    = uvz[0,:]/uvz[2,:]
        v    = uvz[1,:]/uvz[2,:]
        w_l    = (v - intrinsic[1,2])*intrinsic[0,0]*(self.road_xl[0]-self.road_xl[1])/(intrinsic[1,1]*car[0])
        h    = car[2]*car[0]*intrinsic[1,1]/(path[:,2]*(path[:,2]-car[2]))
        valid = (v<img.shape[0])&(v>0)&(path[:,2]-car[2]>0)
        path = path[valid,:]
        u = u[valid]
        v = v[valid]
        w_l = w_l[valid]
        h = h[valid]
        w_l_m    = (self.road_xl[0]+self.road_xl[1])*w_l/(self.road_xl[0]-self.road_xl[1])
        w_r_m    = (self.road_xr[0]+self.road_xr[1])*w_l/(self.road_xl[0]-self.road_xl[1])
        w_r      = (self.road_xr[0]-self.road_xr[1])*w_l/(self.road_xl[0]-self.road_xl[1])
        w_l_diff = w_l_m- w_l
        w_r_diff = w_r_m- w_r
        mask = np.zeros((img.shape[0],img.shape[1]))
        v_last = img.shape[0]
        for i in range(0,len(u)):
            lt  = (int(u[i]-w_l[i]),int(v[i]))
            rl  = (int(u[i]+w_r[i]),int(v[i]+h[i]))
            mask[lt[1]:rl[1],lt[0]:rl[0]]=1

            lt  = (int(u[i]-w_l_m[i]),int(v[i]))
            rl  = (int(u[i]-w_l[i]),int(v_last))
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
            mask_local = np.array([[u/w_l_diff[i] for u in range(0,rl[0]-lt[0])]])
            mask_local = np.repeat(mask_local,rl[1]-lt[1],axis=0)
            mask[up:lo,l:r]=mask_local[shift_up:rl[1]-lt[1]+shift_lo,shift_l:rl[0]-lt[0]+shift_r]

            lt  = (int(u[i]+w_r[i]),int(v[i]))
            rl  = (int(u[i]+w_r_m[i]),int(v_last))
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
            mask_local = np.array([[1-u/w_r_diff[i] for u in range(0,rl[0]-lt[0])]])
            mask_local = np.repeat(mask_local,rl[1]-lt[1],axis=0)
            mask[up:lo,l:r]=mask_local[shift_up:rl[1]-lt[1]+shift_lo,shift_l:rl[0]-lt[0]+shift_r]
            v_last = v[i]       
        return mask

   

    def project_prob(self,img,path_,intrinsic,car):
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

    def road_calculation(self,prob,path):
        road_map_prob,road = self.estimate_road(prob)
        self.road_prob_ml = road_map_prob
        road_l_r = self.road_analysis(road,path)
        mean_l_r = np.mean(road_l_r,0)
        std_l_r = np.std(road_l_r,0)
        self.road_xl =[mean_l_r[0],min(2*std_l_r[0],mean_l_r[0]/2)]
        self.road_xr =[mean_l_r[1],min(2*std_l_r[1],mean_l_r[1]/2)]
        print(self.road_xl,self.road_xr)
        mask_flat = self.project_prob_lr(prob,path,self.intrinsic,self.car)
        mask_flat[mask_flat>0.49] =1
        self.road_prob_flat = mask_flat
        
    def road_analysis(self,road,path):
        x_l_r = np.zeros((road.shape[0],3))
        for i in range(1,path.shape[0]):
            z_s = path[i-1,2]
            z_e = path[i,2]
            x_s = path[i-1,0]
            x_e = path[i,0]
            mask = (road[:,2]>z_s)&(road[:,2]<=z_e)
            road_select = road[mask,0:3]
            x_l = x_s+ (x_e-x_s)*(road_select[:,2] - z_s)/(z_e-z_s) - road_select[:,0]
            x_r = road_select[:,1]- x_s+ (x_e-x_s)*(road_select[:,2] - z_s)/(z_e-z_s)  
            x_l_r[mask,0] = x_l
            x_l_r[mask,1] = x_r
            x_l_r[mask,2] = road[mask,2]
        return (x_l_r)


    # calculate the Most likelihood road region
    def estimate_road(self,prob,center=None):
        prob_new = np.zeros(prob.shape)
        res =[]
        import map_road as mr
        for i in range(prob.shape[0]//2,prob.shape[0]):
            a,b,p = mr.estimate_road(prob[i,:])
            res.append([a,b,p,i,1])
            prob_new[i,a:b]=p

        # project to road plane
        res  = np.array(res)
        res  = res[res[:,2]>0,:]
        road_left = self.intrinsic_inverse@res[:,[True,False,False,True,True]].transpose()
        road_left = self.road_height*road_left/road_left[1,:]
        road_right= self.intrinsic_inverse@res[:,[False,True,False,True,True]].transpose()
        road_right= self.road_height*road_right/road_right[1,:]
        road_left[1,:] = road_right[0,:]

        return prob_new,road_left.transpose()


    def visualize(self,img_bgr):
        import matplotlib.pyplot as plt
        ax1 = plt.subplot(421)
        img_bgr[:,:,1] = 255*self.road_init_mask
        ax1.imshow(img_bgr[:,:,::-1])
        ax1.set_title('drive path')
        ax1.margins(100)
        ax1 = plt.subplot(422)
        img_bgr[:,:,1] = 255*self.unroad_init_mask
        ax1.imshow(img_bgr[:,:,::-1])
        ax1.set_title('road probability')

        ax1 = plt.subplot(426)
        img_bgr[:,:,1] = 255*self.road_prob_ml
        ax1.imshow(img_bgr[:,:,::-1])
        ax1.set_title('drive path')
        ax1.margins(100)
        ax1 = plt.subplot(427)
        img_bgr[:,:,1] = 255*self.road_prob_flat
        ax1.imshow(img_bgr[:,:,::-1])
        ax1.set_title('road probability')
        

        ax1 = plt.subplot(425)
        img_bgr[:,:,1] = 255*self.road_prob
        #ax1.imshow(img_bgr[:,:,::-1])
        ax1.imshow(img_bgr)
        ax1 = plt.subplot(423)  
        ax1.plot(self.road_color_model[0],'b')
        ax1.plot(self.road_color_model[1],'g')
        ax1.plot(self.road_color_model[2],'r')
        ax1 = plt.subplot(424)
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
