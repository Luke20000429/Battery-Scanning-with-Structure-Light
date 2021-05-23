import numpy as np

class Transformer():
    def __init__(self):
        self.src = None
        self.tgt = None
        self.T = None

    def clear(self):
        self.src = None
        self.target = None
        self.T = None

    def pix2point(self, pixs, cloud):
        points = []
        for pix in pixs:
            index = 1944*pix[0,1] + pix[0,0]
            points.append(cloud[index].copy()) 
            # points.append(cloud[1944*pix[0,1] + pix[0,0]-1])
            # points.append(cloud[1944*pix[0,1] + pix[0,0]+1])
            # points.append(cloud[1944*(pix[0,1]-1) + pix[0,0]])
            # points.append(cloud[1944*(pix[0,1]+1) + pix[0,0]])
        return points
        
    def fit(self):
        A = np.hstack([self.src[:], np.ones((self.src.shape[0], 1))])
        re,_,_,_ = np.linalg.lstsq(A, self.tgt, rcond=None)
        self.T = re
        return re

    def transform(self, cloud):
        A = np.hstack([cloud, np.ones((cloud.shape[0], 1))])
        A = A.astype(np.float32)
        transCloud = A@self.T
        return transCloud
        
