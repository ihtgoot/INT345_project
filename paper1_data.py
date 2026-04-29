import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

'''
clean image → synthetic turbulence → distorted image

Pipeline:
    generate displacement field
    warp image
    apply spatial blur
    (optional) temporal variation
    add noise
        
        image = load_clean_image()
        dx, dy = generate_displacement(H, W)+
        warped = warp_image(image, dx, dy)
        variation = compute_variation(dx, dy)
        blurred = apply_spatial_blur(warped, variation)
        final = add_noise(blurred)
'''

def Normalize(x):
    m = x.min()
    M = x.max()
    if M - m == 0:
        return x * 0   
    n = (x - m) / (M - m)
    n = 2 * n - 1
    return n

def GenerateDisplacement(H,W):
    '''
    δ(x, y) = (dx, dy)
        dx = horizontal shift
        dy = vertical shift
    
    Random noise
    smooth it
    normalize
    '''

    n1x, n1y = np.random.randn(H, W), np.random.randn(H, W)  # large
    n2x, n2y = np.random.randn(H, W), np.random.randn(H, W)  # medium
    n3x, n3y = np.random.randn(H, W), np.random.randn(H, W)  # small 

    # large scale (smooth)
    s1x = cv.GaussianBlur(n1x, (0,0), 25)
    s1y = cv.GaussianBlur(n1y, (0,0), 25)
    s2x = cv.GaussianBlur(n2x, (0,0), 8)
    s2y = cv.GaussianBlur(n2y, (0,0), 8)
    s3x = cv.GaussianBlur(n3x, (0,0), 2)   
    s3y = cv.GaussianBlur(n3y, (0,0), 2)

    dx = 4*Normalize(s1x) + 3*Normalize(s2x) + 2*Normalize(s3x)
    dy = 4*Normalize(s1y) + 3*Normalize(s2y) + 2*Normalize(s3y)

    return dx, dy


def Warp(image, dx, dy, scalinig):
    H, W = image.shape[:2]

    x, y = np.meshgrid(np.arange(W), np.arange(H))

    map_x = (x + dx*scalinig).astype(np.float32)
    map_y = (y + dy*scalinig).astype(np.float32)

    warped = cv.remap(image, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)

    return warped


def ComputeVariation(dx,dy):
    dx_y, dx_x = np.gradient(dx)
    dy_y, dy_x = np.gradient(dy)
    variation = np.sqrt(dx_x**2 + dx_y**2 + dy_x**2 + dy_y**2)
    return Normalize(variation)

def ApplyBlur(image, variation) -> np.uint8 :
    v = (variation - variation.min()) / (variation.max() - variation.min() + 1e-8)
    blur_weak   = cv.GaussianBlur(image, (0,0), 1)
    blur_medium = cv.GaussianBlur(image, (0,0), 3)
    blur_strong = cv.GaussianBlur(image, (0,0), 6)

    v = v[..., None]

    weak_mask   = (1 - v)
    medium_mask = 4 * v * (1 - v)
    strong_mask = v
    sum_mask = weak_mask + medium_mask + strong_mask + 1e-8
    weak_mask   /= sum_mask
    medium_mask /= sum_mask
    strong_mask /= sum_mask

    output = (
        blur_weak * weak_mask +
        blur_medium * medium_mask +
        blur_strong * strong_mask
    )

    return output.astype(np.uint8)

def NewDisplacemnt (dy, dx, img):
    '''
     dx_t = new_smooth_noise (same sigma, slightly different seed)

    '''
    H, W, c = img.shape
    ndx, ndy = GenerateDisplacement(H,W)
    dy_t = dy + ndy 
    dx_t = dx + ndx
    return dx_t,dy_t
    

def AddNoise(image, sigma=5):
    noise = np.random.randn(*image.shape) * sigma
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)



def ApplySpatialEnvelope(dx, dy, H, W):
    # ----- Step 1: random smooth field -----
    noise = np.random.randn(H, W).astype(np.float32)
    smooth = cv.GaussianBlur(noise, (0, 0), sigmaX=1, sigmaY=0.1)
    # normalize to [0,1]
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    # ----- Step 2: random center bias -----
    cx = np.random.uniform(0.3 * W, 0.7 * W)
    cy = np.random.uniform(0.3 * H, 0.7 * H)
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    sigma = np.random.uniform(0.2 * min(H, W), 0.5 * min(H, W))
    radial = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    # normalize radial to [0,1]
    radial = (radial - radial.min()) / (radial.max() - radial.min() + 1e-8)
    # ----- Step 3: combine -----
    A = smooth * radial
    # normalize again (important)
    A = (A - A.min()) / (A.max() - A.min() + 1e-8)
    # ----- Step 4: apply to displacement -----
    dx_mod = dx * A
    dy_mod = dy * A
    return dx_mod, dy_mod, A



def main(path):
    img = cv.imread(path)  # grayscale

    H, W , c= img.shape

    dx, dy = GenerateDisplacement(H, W)
    dx, dy, envelope = ApplySpatialEnvelope(dx, dy, H, W)
    out1 = Warp(img, dx, dy, 2)
    variation=Normalize(ComputeVariation(dx, dy))
    blured1 = ApplyBlur(out1,variation)

    # for i in range(layer):

    nx, ny = NewDisplacemnt(dx,dy,blured1)
    
    outN = Warp(blured1, nx, ny, 2.5)
    varitionN = Normalize(ComputeVariation(nx,ny))
    bluredN = ApplyBlur(outN,varitionN)
    final = AddNoise(bluredN)

    # cv.imshow("dx", Normalize(dx))
    # cv.imshow("dy", Normalize(dy))
    # cv.imshow("variation", variation)

    cv.imshow("original", img)
    # cv.imshow("cv warp", out1)
    # cv.imshow("blured1", blured1)
    # cv.imshow("cv warp2", outN)
    # cv.imshow("blured", bluredN)
    cv.imshow("final", final)
    cv.imshow("env", envelope)


    cv.waitKey(0)



if __name__ == "__main__":
    main(path="/home/ihtgoot/CV/detailed-shot-of-ripples-at-sunset-free-image.jpeg")