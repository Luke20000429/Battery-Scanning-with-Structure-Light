import numpy as np

class Transformer():
    def __init__(self):
        self.src = None
        self.tgt = None
        self.T = np.eye(4)

    def clear(self):
        self.src = None
        self.target = None
        self.T = np.eye(4)

    def TAGpix2point(self, pixs, cloud):
        # tag pix shape (n,1,2)
        points = []
        for pix in pixs:
            # pix[0,1] in (0,1472)
            # pix[0,0] in (0,1944)
            index = 1944*pix[0,1] + pix[0,0]
            points.append(cloud[index].copy()) 
            # points.append(cloud[1944*pix[0,1] + pix[0,0]-1])
            # points.append(cloud[1944*pix[0,1] + pix[0,0]+1])
            # points.append(cloud[1944*(pix[0,1]-1) + pix[0,0]])
            # points.append(cloud[1944*(pix[0,1]+1) + pix[0,0]])
        return points

    def pix2point(self, pix, cloud):
        # pix shape (n,2)
        index = 1944*pix[:,1] + pix[:,0]
        points = cloud[index].copy()
        return points
        
    def fit(self):
        A = np.hstack([self.src[:], np.ones((self.src.shape[0], 1))])
        re,_,_,_ = np.linalg.lstsq(A, self.tgt, rcond=None)
        self.T = self.T@np.hstack((re, np.array([[0],[0],[0],[1]])))
        return re

    def transform(self, cloud):
        A = np.hstack([cloud, np.ones((cloud.shape[0], 1))])
        A = A.astype(np.float32)
        transCloud = A@self.T
        return transCloud[:, :3]
        
