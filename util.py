import cv2
import numpy as np

#yeah yeah
_bits = []
for i in range(256):
    bs = bin(i)[2:].rjust(8,'0')
    _bits.append(np.array([float(v) for v in bs]))

def homography_transform(X, H):
    '''
    Perform homography transformation on a set of points X
    using homography matrix H
    
    Input - X: a set of 2D points in an array with size (N,2)
            H: a 3*3 homography matrix
    Output -Y: a set of 2D points in an array with size (N,2)
    '''
    X_homogeneous = np.hstack([X,np.ones((X.shape[0],1))])
    Y = np.dot(H,X_homogeneous.T).T
    return Y[:,:2] / Y[:,2][:,None]

def get_match_points(kp1, kp2, matches):
    '''
    Returns list of paired keypoint locations
        
    Input - kp1: Keypoint matrix 1 of shape (N,4)
            kp2: Keypoint matrix 1 of shape (M,4)
            matches: List of matching pairs indices between the 2 sets of keypoints (K,2)
    
    Output - An array of shape (K,4) where row i contains pixel locations corresponding
             to a matched keypoint in the 2 images : [img1_x, img1_y, img2_x, img2_y]
    '''
    return np.hstack([kp1[matches[:,0],:2], kp2[matches[:,1],:2]])

def kps_to_matrix(kps):
    '''
    Converts cv2 container of keypoint locations into numpy array
        
    Input - kps: opencv container of keypoints location
    
    Output - K: A numpy Keypoint matrix of shape (N,4)
    '''
    K = np.zeros((len(kps),4))
    for i in range(len(kps)):
        K[i,:2] = kps[i].pt
        K[i,2] = kps[i].angle
        K[i,3] = kps[i].octave
    return K

def expand_binarize(desc):
    '''
    Explicitly expand packed binary keypoint descriptors like AKAZE and ORB.
    You do not need to modify or worry about this.

    AKAZE and ORB return a descriptor that is binary. Usually one compares
    descriptors using the hamming distance (# of bits that differ). This is
    usually fast since one can do this with binary operators. On Intel
    processors, there's an instruction for this: popcnt/population count.

    On the other hand, this prevents you from actually implementing all the steps
    of the pipeline and requires you writing a hamming distance. So instead, we
    explicitly expand the feature from F packed binary uint8s to (8F) explicit 
    binary 0 or 1 descriptors. The square of the L2 distance of these
    descriptors is the hamming distance.
    
    Converts a matrix where each row is a vector containing F uint8s into their
    explicit binary form.
        
    Input - desc: matrix of size (N,F) containing N 8F dimensional binary
                  descriptors packed into N, F dimensional uint8s
    
    Output - binary_desc: matrix of size (N,8F) containing only 0s or 1s that 
                          expands this to be explicit
    '''
    N, F = desc.shape
    binary_desc = np.zeros((N,F*8))
    for i in range(N):
        for j in range(F):
            binary_desc[i,(j*8):((j+1)*8)] = _bits[desc[i,j]]
    return binary_desc

def get_AKAZE(I):
    '''
    Extracts AKAZE keypoints and descriptors from an image
        
    Input - img: Input image of shape (H,W,3)
    
    Output - kps: (K,4) matrix where each row is [x,y,angle,octave]
             desc: (K,1024) matrix of AKAZE descriptors expanded to be
                   comparable using squared L2 distance
    '''
    akaze = cv2.AKAZE_create()
    kps, D = akaze.detectAndCompute(I, None)
    return kps_to_matrix(kps), expand_binarize(D).astype(np.float32)

def distance(XY, M):
    '''
    Given a set of N correspondences XY of the form [x,y,x',y'], and estimated transform_matrix M
    calculate the distance between M [x,y,1] and [x',y',1].
    
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
            M: a (3,3) homography matrix that (if the correspondences can be
            described by a homography) satisfies [x',y',1]^T === M [x,y,1]^T
    '''
    k = XY.shape[0]
    origin = np.hstack([XY[:, :2], np.ones((k,1))])
    target = XY[:, 2:]
    Tx = M@origin.T
    dist = target - (Tx[:2]/Tx[2]).T
    return np.linalg.norm(dist, axis=1)

def fit_homography(XY):
    '''
    Given a set of N correspondences XY of the form [x,y,x',y'],
    fit a homography from [x,y,1] to [x',y',1].
    
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
    Output -H: a (3,3) homography matrix that (if the correspondences can be
            described by a homography) satisfies [x',y',1]^T === H [x,y,1]^T

    '''
    k = XY.shape[0]
    p = XY[:, :2]
    p = np.hstack([p, np.ones((k,1))])
    p = np.hstack([p,p]).reshape(-1,3)
    A1 = np.hstack([np.zeros((k, 1)), np.ones((k, 1))]).reshape(-1,1)*p
    A2 = np.hstack([-np.ones((k, 1)), np.zeros((k, 1))]).reshape(-1,1)*p
    b = XY[:, :1:-1]
    b = b*[1,-1]
    b = b.reshape(-1,1)
    A3 = b*p
    A = np.hstack([A1, A2, A3])
    l, v = np.linalg.eigh(A.T@A)
    H = v[:, abs(l).argmin()]
    H = H.reshape((3,3))
    H = H/H[-1,-1]
    return H

def RANSAC_fit_homography(XY, eps=1, nIters=1000):
    '''
    Perform RANSAC to find the homography transformation 
    matrix which has the most inliers
        
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
            eps: threshold distance for inlier calculation
            nIters: number of iteration for running RANSAC
    Output - bestH: a (3,3) homography matrix fit to the 
                    inliers from the best model.

    Note:
    a) It sample 4 
    '''
    bestH, bestCount = np.eye(3), -1
    bestCount = -1
    k = XY.shape[0]
    for it in range(nIters):
        select = np.random.choice(k, size=4, replace=False)
        H = fit_homography(XY[select, :])
        dist = distance(XY, H)
        E = (dist < eps).sum()
        if E > bestCount:
            bestH = H
            bestCount = E
    print(bestCount/k)
    return bestH