import numpy as np
import open3d as o3d
import copy
from matplotlib import pyplot as plt
from readData import *

class MyPcd():
    def __init__(self, filename):
        self.points = np.load(filename)
        self.pcd = o3d.geometry.PointCloud()
        
    def filterPoints(self, prange=(2000, 1500, (-1000, 1000))):
        self.mask = ((np.abs(self.points[:,0])<=prange[0]) & (np.abs(self.points[:,1])<=prange[1]) 
                            & (self.points[:,2]>=prange[2][0]) & (self.points[:,2]<=prange[2][1]))

    def np2pcd(self, c=None):
        self.pcd.points = o3d.utility.Vector3dVector(self.points[self.mask, :3])
        if c:
            if c.ndim == 2 and c.shape[1] == 3:
                self.pcd.colors = o3d.utility.Vector3dVector(c)
            else:
                c = np.stack([c,c,c], 1)
                self.pcd.colors = o3d.utility.Vector3dVector(c)
        elif self.points.shape[1] == 6:
            self.pcd.colors = o3d.utility.Vector3dVector(self.points[self.mask, 3:])
        elif self.points.shape[1] == 4:
            c = self.points[self.mask, 3]
            c = np.stack([c,c,c], 1)
            self.pcd.colors = o3d.utility.Vector3dVector(c)

    def downSample(self, voxel_size=0.1, inplace=True):
        if inplace:
            self.pcd = self.pcd.voxel_down_sample(voxel_size)
        else:
            return self.pcd.voxel_down_sample(voxel_size)

    def denoise_radius(self, nb_points=5, radius=0.3, inplace=False):
        cl, indice = self.pcd.remove_radius_outlier(nb_points, radius)
        print(np.asarray(cl.points).shape)
        if inplace:
            self.pcd = cl
        else:
            return cl

    def denoise_statistical(self, nb_neighbors=50, std_ratio=0.001, inplace=False):
        cl, indice = self.pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        print(np.asarray(cl.points).shape)
        if inplace:
            self.pcd = cl
        else:
            return cl

    def denoise_wmp(self):
    
        pass

    def getPoints(self, pcd=None):
        if pcd:
            if pcd.has_points():
                return np.asarray(pcd.points)
            else:
                return None
        else:
            if self.pcd.has_points():
                return np.asarray(self.pcd.points)
            else:
                return None

    def getColors(self, pcd=None):
        if pcd:
            if pcd.has_colors():
                return np.asarray(pcd.colors)
            else:
                return None
        else:
            if self.pcd.has_colors():
                return np.asarray(self.pcd.colors)
            else:
                return None

    def getNormals(self, pcd=None):
        if pcd:
            if pcd.has_normals():
                return np.asarray(pcd.normals)
            else:
                pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=10))
                return np.asarray(pcd.normals)
        else:
            if self.pcd.has_normals():
                return np.asarray(self.pcd.normals)
            else:
                self.pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=10))
                return np.asarray(self.pcd.normals)

    def pcd2np(self, pcd=None, gray=True):
        if pcd:
            if pcd.has_colors():
                if gray:
                    points = np.hstack([self.getPoints(pcd), self.getColors(pcd)[:,:1]])
                else:
                    points = np.hstack([self.getPoints(pcd), self.getColors(pcd)])
            else:
                points = self.getPoints(pcd)
        else:
            if self.pcd.has_colors():
                if gray:
                    points = np.hstack([self.getPoints(), self.getColors()[:,:1]])
                else:
                    points = np.hstack([self.getPoints(), self.getColors()])
            else:
                points = self.getPoints()

        return points



