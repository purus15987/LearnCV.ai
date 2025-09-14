import cv2
import numpy as np

# ------------- Color conversions -------------
def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def to_bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def to_hsv(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

def to_ycrcb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

# ------------- Geometric transforms -------------
def rotate(img, angle_deg, center=None, scale=1.0):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    return cv2.warpAffine(img, M, (w, h))

def scale(img, fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)

def translate(img, tx=0, ty=0):
    (h, w) = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))

def affine_transform(img, src_pts, dst_pts):
    M = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))
    (h, w) = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))

def perspective_transform(img, src_pts, dst_pts):
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    (h, w) = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h))

# ------------- Filtering -------------
def mean_filter(img, ksize=3):
    return cv2.blur(img, (ksize, ksize))

def gaussian_filter(img, ksize=3, sigma=0):
    k = (ksize | 1)  # ensure odd
    return cv2.GaussianBlur(img, (k, k), sigma)

def median_filter(img, ksize=3):
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(img, k)

# ------------- Edges -------------
def sobel_edges(img_gray):
    x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(x, y)
    mag = np.uint8(255 * (mag / (mag.max() + 1e-8)))
    return mag

def laplacian_edges(img_gray):
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    return lap

def canny_edges(img_gray, t1=100, t2=200):
    return cv2.Canny(img_gray, t1, t2)

# ------------- Morphology -------------
def morphology(img, op="dilate", ksize=3, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    if op == "dilate":
        return cv2.dilate(img, kernel, iterations=iterations)
    if op == "erode":
        return cv2.erode(img, kernel, iterations=iterations)
    if op == "open":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if op == "close":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return img

# ------------- Enhancement -------------
def histogram_equalization(img_bgr):
    if len(img_bgr.shape) == 2 or img_bgr.shape[2] == 1:
        return cv2.equalizeHist(img_bgr)
    # Apply CLAHE on each channel in LAB for color images
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def contrast_stretch(img):
    # Normalize to full [0,255]
    out = np.empty_like(img)
    if len(img.shape) == 2:
        minv, maxv = np.min(img), np.max(img)
        out = ((img - minv) * 255.0 / (maxv - minv + 1e-8)).astype(np.uint8)
    else:
        for c in range(img.shape[2]):
            ch = img[..., c]
            minv, maxv = np.min(ch), np.max(ch)
            out[..., c] = ((ch - minv) * 255.0 / (maxv - minv + 1e-8)).astype(np.uint8)
    return out

def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

# ------------- Bitwise ops -------------
def bitwise_and(img1, img2):
    img2r = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.bitwise_and(img1, img2r)

def bitwise_or(img1, img2):
    img2r = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.bitwise_or(img1, img2r)

def bitwise_xor(img1, img2):
    img2r = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.bitwise_xor(img1, img2r)

def bitwise_not(img):
    return cv2.bitwise_not(img)

# ------------- Utilities -------------
def half_split_comparison(original, processed):
    h, w = original.shape[:2]
    proc_resized = cv2.resize(processed, (w, h))
    half = w // 2
    left = original[:, :half]
    right = proc_resized[:, half:]
    return np.hstack([left, proc_resized[:, half:]])
