import numpy as np
import matplotlib.pyplot as plt
import vocorrection as vc

# visualization path and road edge
# @input path: path point nx3
# @road  function ax^2+bx+c
# @cam   [cols,rows,fx,fy,cy,cy,height,width]
def visualization(path,road,road_2,cam):
    road_y = np.array(list(range(0,cam[0])))
    road_x = road[0]*road_y*road_y+road[1]*road_y+road[2]


    road_valid = (road_y>0)&(road_y<cam[1])
    plt.plot(road_x[road_valid],road_y[road_valid],'r')

    road_x_2 = road_2[0]*road_y*road_y+road_2[1]*road_y+road_2[2]
    plt.plot(road_x_2[road_valid],road_y[road_valid],'r')

    path_2d = path[:,0:2].copy()
    path_2d[:,0] = cam[2]*path[:,0]/path[:,2]+cam[4]
    path_2d[:,1] = cam[3]*(path[:,1]+cam[6])/path[:,2]+cam[5]
    plt.scatter(path_2d[:,0],cam[1]-path_2d[:,1])


def visualization_remap(path,path_c,road,road_2,cam):
    fig, ax = plt.subplots()
    road_y = np.array(list(range(0,cam[0])))
    road_x = road[0]*road_y*road_y+road[1]*road_y+road[2]
    
    path_from = np.min(path[:,2])
    path_to = np.max(path[:,2])
    
    road_Z = cam[6]*cam[3]/(road_y-cam[5])
    road_X = road_Z*(road_x -cam[4])/cam[2]
    valid = (road_Z>path_from-5)&(road_Z<path_to+5)
    ax.plot(road_X[valid],road_Z[valid],'y')

    road_x_2 = road_2[0]*road_y*road_y+road_2[1]*road_y+road_2[2]

    road_Z = cam[6]*cam[3]/(road_y-cam[5])
    road_X = road_Z*(road_x_2 -cam[4])/cam[2]
    valid = (road_Z>path_from-5)&(road_Z<path_to+5)
    ax.plot(road_X[valid],road_Z[valid],'y',label='road edge')
    
    ax.plot(path[:,0],path[:,2],'r',label='initial path')
    ax.plot(path_c[:,0],path_c[:,2],'g',label='corrected path')
    legend = ax.legend(loc='lower left', shadow=True, fontsize='x-large')
    plt.xlabel('X/m')
    plt.ylabel('Z/m')
    plt.title('Path Correction')
    #plt.scatter(path[:,0],path[:,2],'r')
    #plt.scatter(path_c[:,0],path_c[:,2],'g')



# remap road function when it is linear
def linear_map(b,c,fx,fy,cx,cy,Y):
    b_new = (b*cy+c-cx)/fx
    c_new = b*Y*fy/fx
    return b_new,c_new
def inverse_linear_map(b,c,fx,fy,cx,cy,Y):
    b_new = (c*fx)/(Y*fy)
    c_new = b*fx+cx-c*fx*cy/(Y*fy)
    return b_new,c_new


def main():
    vco = vc.EgoMotionCorrection(320,320,320,240,4,1.75)

    path = np.zeros((30,3))
    path[:,2] = np.array(list(range(1,31)))
    path[:,0] = 0.1*np.array(list(range(1,31)))-2
    path[:,1] = 1.75*np.ones((path.shape[0]))
    path[5:20:3,0] = 1
    lamdas = 10*np.ones(30)
    road =[0,1,80]
    b,c = linear_map(1,80,320,320,320,240,1.75)
    b_n,c_n = inverse_linear_map(b,c-10,320,320,320,240,1.75)
    road_2 =[0,b_n,c_n]
    cam  =[640,480,320,320,320,240,1.75]

    path_c = vco.correct(path,lamdas,0,1,80)
    print(path_c)
    visualization_remap(path,np.array(path_c),road,road_2,cam)

    plt.show()


if __name__ == '__main__':
    main()
