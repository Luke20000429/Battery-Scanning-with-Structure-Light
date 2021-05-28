import numpy as np
from util import *
import cv2
from matplotlib import pyplot as plt

from readData import *
from showFig import *
from showCloud import *
from warp import *

from transform import Transformer

class Register():
    def __init__(self):
        self.tf = Transformer()
        self.src = None
        self.tgt = None
        self.points0 = None
        self.points1 = None
        self.mat0 = None
        self.mat1 = None

    def clear(self):
        self.tf.clear()
        self.src = None
        self.tgt = None
        self.points0 = None
        self.points1 = None
        self.mat0 = None
        self.mat1 = None

    def loadData(self, names):
        self.points0, self.mat0 = getPoints(names[0])
        self.points1, self.mat1 = getPoints(names[1])

    def getMatch(self, eps=0.5, region=900, debug=False):
        # mat1 must under mat1
        # scroll down mat0 -> mat1 
        depth0 = self.points0[:,2].reshape((1472, 1944))
        depth1 = self.points1[:,2].reshape((1472, 1944))

        kp1, desc1 = get_AKAZE(depth0[:region])
        kp2, desc2 = get_AKAZE(depth1[-region:])
        matches = find_matches(desc1, desc2, eps)
        floatpix = get_match_points(kp1, kp2, matches)

        # 2 options
        # points = np.around(floatpoint)
        pix = np.int32(floatpix)

        pix[:, 3] += 1472 - region
        if debug:
            print(pix.shape)
            plt.imshow(self.mat0, 'gray')
            plt.scatter(pix[:, 0], pix[:, 1])
            plt.show()

            plt.imshow(self.mat1, 'gray')
            plt.scatter(pix[:, 2], pix[:, 3])
            plt.show()

        self.tgt = self.tf.pix2point(pix[:, :2], self.points0)
        self.src = self.tf.pix2point(pix[:, 2:], self.points1)

        # return self.tgt, self.src

    def register(self, color=True):
        self.tf.tgt = self.tgt
        self.tf.src = self.src
        self.tf.fit()
        self.points1 = self.tf.transform(self.points1)
        ## colorize clouds
        if color:
            self.points0 = addColor(self.points0, self.mat0)
            self.points1 = addColor(self.points1, self.mat1)

        self.tf.clear()

        return self.points0, self.points1

        


