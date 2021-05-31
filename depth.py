import numpy as np
import scipy.ndimage


def getDepth(points):
    """
    Reshape z dim to image
    """

    if points.shape[0] != 1472*1944:
        print("require full points cloud!")
    
    depth = points[:,2].reshape((1472, 1944))

    return depth

def preprocDepth(depth, h=(-100,-30)):
    """
    Set z beyond range to avg height
    """

    avg = depth.mean()
    depth[depth>h[1]] = avg
    depth[depth<h[0]] = avg

    return depth

def gradDepth(depth):
    """
    Run preprocess before getting gradient
    """

    Sx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Sy = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    # Guassian + gradient filter

    Gx = scipy.ndimage.convolve(depth, Sx, mode='constant')
    Gy = scipy.ndimage.convolve(depth, Sy, mode='constant')

    grad_magnitude = np.sqrt(Gx**2+Gy**2)

    norm_grad = (grad_magnitude-grad_magnitude.mean())/(grad_magnitude.max()-grad_magnitude.min())

    return norm_grad



