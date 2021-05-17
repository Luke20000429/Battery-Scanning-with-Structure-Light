import numpy as np
from util import *
import os
import cv2


def compute_distance(desc1, desc2):
    '''
    Calculates L2 distance between 2 binary descriptor vectors.
        
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
    
    Output - dist: a (N,M) L2 distance matrix where dist(i,j)
             is the squared Euclidean distance between row i of 
             desc1 and desc2. You may want to use the distance
             calculation trick
             ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y
    '''
    # desc = desc1.reshape(desc1.shape[0],1,desc1.shape[1]) - desc2
    # dist = (desc**2).sum(axis=2)
    dist1 = (desc1**2).sum(axis=1)
    dist2 = (desc2**2).sum(axis=1)
    dist = dist1.reshape(-1,1) + dist2 - 2*desc1@desc2.T
    return dist

def find_matches(desc1, desc2, ratioThreshold):
    '''
    Calculates the matches between the two sets of keypoint
    descriptors based on distance and ratio test.
    
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
            ratioThreshhold : maximum acceptable distance ratio between 2
                              nearest matches 
    
    Output - matches: a list of indices (i,j) 1 <= i <= N, 1 <= j <= M giving
             the matches between desc1 and desc2.
             
             This should be of size (K,2) where K is the number of 
             matches and the row [ii,jj] should appear if desc1[ii,:] and 
             desc2[jj,:] match.
    '''
    dist = compute_distance(desc1, desc2)
    sort_dist = dist.copy()
    sort_dist.sort(axis=1)
    ratio = sort_dist[:, 0]/sort_dist[:, 1]
    match1 = np.arange(dist.shape[0])[ratio < ratioThreshold]
    match2 = dist.argmin(axis=1)[ratio < ratioThreshold]
    matches = np.hstack([match1.reshape(-1,1), match2.reshape(-1,1)])
    # print(matches.shape)
    return matches

def draw_matches(img1, img2, kp1, kp2, matches):
    '''
    Creates an output image where the two source images stacked vertically
    connecting matching keypoints with a line. 
        
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            kp1: Keypoint matrix for image 1 of shape (N,4)
            kp2: Keypoint matrix for image 2 of shape (M,4)
            matches: List of matching pairs indices between the 2 sets of 
                     keypoints (K,2)
    
    Output - Image where 2 input images stacked vertically with lines joining 
             the matched keypoints

    Hint: see cv2.line
    '''
    #Hint:
    #Use common.get_match_points() to extract keypoint locations
    points = np.int32(get_match_points(kp1, kp2, matches))
    output = np.vstack([img1, img2])
    offset = img1.shape[0]
    # p1 = points[:, :2]
    # p2 = points[:, 2:]
    points[:, 3] += offset
    # print(points.shape)
    for point in points:
        output = cv2.line(output, tuple(point[:2]), tuple(point[2:]), (255,0,0), 1)
    return output


def warp_and_combine(img1, img2, H):
    '''
    You may want to write a function that merges the two images together given
    the two images and a homography: once you have the homography you do not
    need the correspondences; you just need the homography.

    Writing a function like this is entirely optional, but may reduce the chance
    of having a bug where your homography estimation and warping code have odd
    interactions.
    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            H: homography mapping betwen them

    Output - V: stitched image of size (?,?,3); unknown since it depends on H
    '''
    pattern1 = np.ones(img1.shape[:2])
    pattern2 = np.ones(img2.shape[:2])

    corners = np.float32([(0, 0, 1), (img1.shape[1], 0, 1), (img1.shape[1], img1.shape[0], 1), ((0, img1.shape[0], 1))])
    Tcorners = H@corners.T
    Tcorners = Tcorners[:2]/Tcorners[2]
    offset = np.int32((np.minimum((0,0), Tcorners.min(axis=1)), np.maximum(img2.shape[1::-1], Tcorners.max(axis=1))))
    imsize = tuple(offset[1]-offset[0])
    T = np.eye(3)
    T[:2, 2] -= offset[0].T

    I1 = cv2.warpPerspective(img1, T@H, imsize)
    I2 = cv2.warpPerspective(img2, T, imsize)
    p1 = cv2.warpPerspective(pattern1, T@H, imsize)
    p2 = cv2.warpPerspective(pattern2, T, imsize)
    pattern = p1 + p2
    p = pattern.copy()
    pattern[pattern<=0] = 1
    V = (np.float32(I1)+np.float32(I2))/pattern.reshape(pattern.shape[0], pattern.shape[1])
    return V, p


def make_warped(img1, img2):
    '''
    Take two images and return an image, putting together the full pipeline.
    Transform img1 to stitch with img2
    You should return an image of the panorama put together.

    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 1 of shape (H2,W2,3)
            H: a (3,3) homography matrix
    
    Output - Final stitched image

    Be careful about:
    a) H is applied on img1
    '''
    kp1, desc1 = get_AKAZE(img1)
    kp2, desc2 = get_AKAZE(img2)
    matches = find_matches(desc2, desc1, 0.7)
    points = np.int32(get_match_points(kp2, kp1, matches))
    H = RANSAC_fit_homography(points)

    stitched, pattern = warp_and_combine(img2, img1, H)
    
    return np.uint8(stitched), pattern, H
