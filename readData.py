import numpy as np
import sys 

def getPoints(name, prange=None, skip=0):
    print("read in " + name)
    mat = None
    if name[-4:] == ".npy":
        points = np.load(name)
        mat = points[:, 3].astype('uint8')
        mat = mat.reshape((1472, 1944))
        points = points[:,:3]
    else:
        points = np.loadtxt(name)
    print("x range: %d ~ %d"%(points[:,0].min(), points[:,0].max()))
    print("y range: %d ~ %d"%(points[:,1].min(), points[:,1].max()))
    print("z range: %d ~ %d"%(points[:,2].min(), points[:,2].max()))
    if prange:
        # points = points[(np.abs(points[:,0])<=prange[0]) & (np.abs(points[:,1])<=prange[1]) & (np.abs(points[:,2])<=prange[2])]
        points = points[(np.abs(points[:,0])<=prange[0]) & (np.abs(points[:,1])<=prange[1]) &
                        (points[:,2]>=prange[2][0]) & (points[:,2]<=prange[2][1])]
    if skip:
        point_range = range(0, points.shape[0], skip) # skip points to prevent crash
        points = points[point_range]
    return points, mat

def filterPoints(points, prange=None, skip=0):
    if prange:
        # points = points[(np.abs(points[:,0])<=prange[0]) & (np.abs(points[:,1])<=prange[1]) & (np.abs(points[:,2])<=prange[2])]
        points = points[(np.abs(points[:,0])<=prange[0]) & (np.abs(points[:,1])<=prange[1]) &
                        (points[:,2]>=prange[2][0]) & (points[:,2]<=prange[2][1])]
    if skip:
        point_range = range(0, points.shape[0], skip) # skip points to prevent crash
        points = points[point_range]
    return points

if __name__ == '__main__':
    getPoints("..\dataset\scan7.npy", prange=(200, 150, 100), skip=20)
    