import numpy as np
import sys 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visual(name='sample.txt', prange=(200, 150, 100), skip=20):
    points = np.loadtxt(name)
    print("x range: %d ~ %d"%(points[:,0].min(), points[:,0].max()))
    print("y range: %d ~ %d"%(points[:,1].min(), points[:,1].max()))
    print("z range: %d ~ %d"%(points[:,2].min(), points[:,2].max()))
    points = points[(np.abs(points[:,0])<=prange[0]) & (np.abs(points[:,1])<=prange[1]) & (np.abs(points[:,2])<=prange[2])]
    point_range = range(0, points.shape[0], skip) # skip points to prevent crash
    x = points[point_range, 0]
    y = points[point_range, 1]
    z = points[point_range, 2]
    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,   # x
            y,   # y
            z,   # z
            s=0.1,
            c=z, # height data for color
            cmap='viridis',
            marker="o")
#     ax.axis('equal')  # {equal, scaled}
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('feature visulization')
    # ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))  
    # ax.axis('off')          # 设置坐标轴不可见
    ax.grid(False)          # 设置背景网格不可见
    # plt.savefig('3D scatter plot.png')  #保存为图片
    plt.show()
    return 

if __name__ == '__main__':
    visual('sample.txt', skip=20)
    