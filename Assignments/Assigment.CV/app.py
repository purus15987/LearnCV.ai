import streamlit as st
import cv2
import numpy as np
from pathlib import Path

st.set_page_config(layout="wide", page_title="Image Processing Toolkit", initial_sidebar_state="expanded")

def to_bytes(img, ext='.png'):
    is_success, buffer = cv2.imencode(ext, img)
    return buffer.tobytes()

def get_image_info(image, image_bytes=None, filename=""):
    h, w = image.shape[:2]
    c = 1 if image.ndim==2 else image.shape[2]
    size = None
    dpi = "N/A"
    fmt = Path(filename).suffix[1:].upper() if filename else "N/A"
    if image_bytes:
        size = len(image_bytes)
    return {"height":h, "width":w, "channels":c, "dpi":dpi, "format":fmt, "size_bytes":size}

st.title("🖼️ Image Processing Toolkit — Streamlit + OpenCV")

menu = st.sidebar.selectbox("Menu", ["File", "About"])
if menu == "About":
    st.sidebar.markdown("Upload an image and select operations from the sidebar. Built with Streamlit & OpenCV.")

st.sidebar.header("File")
uploaded = st.sidebar.file_uploader("Open → Upload an image", type=['png','jpg','jpeg','bmp','tiff'])
save_btn = st.sidebar.button("Save processed image")

st.sidebar.header("Image Info (Toggle to show)")
show_info = st.sidebar.checkbox("Show image info", value=True)

st.sidebar.header("Color Conversions")
col_conv = st.sidebar.selectbox("Choose color conversion", ["None","RGB ↔ BGR","RGB ↔ HSV","RGB ↔ YCrCb","Grayscale"])

st.sidebar.header("Transformations")
rot_angle = st.sidebar.slider("Rotation angle (degrees)", -180, 180, 0)
scale = st.sidebar.slider("Scaling factor", 10, 300, 100)
tx = st.sidebar.slider("Translate X (pixels)", -200, 200, 0)
ty = st.sidebar.slider("Translate Y (pixels)", -200, 200, 0)
do_affine = st.sidebar.checkbox("Apply Affine Transform")
do_perspective = st.sidebar.checkbox("Apply Perspective Transform")

st.sidebar.header("Filtering & Morphology")
filter_type = st.sidebar.selectbox("Smoothing / Edge", ["None","Gaussian","Median","Mean","Sobel","Laplacian"])
kernel_size = st.sidebar.slider("Kernel size (odd)", 1, 31, 3, step=2)
morph_op = st.sidebar.selectbox("Morphology", ["None","Dilation","Erosion","Opening","Closing"])
morph_iter = st.sidebar.slider("Morph iterations", 1, 10, 1)

st.sidebar.header("Enhancement & Edge Detection")
enhance = st.sidebar.selectbox("Enhancement", ["None","Histogram Equalization","Contrast Stretching","Sharpen"])
edge_choice = st.sidebar.selectbox("Edge Detection", ["None","Canny","Sobel","Laplacian"])
canny_thresh1 = st.sidebar.slider("Canny Thresh1", 0, 500, 100)
canny_thresh2 = st.sidebar.slider("Canny Thresh2", 0, 500, 200)

st.sidebar.header("Compression")
save_format = st.sidebar.selectbox("Save format", ["PNG",".png",".jpg",".jpg",".bmp"])
compare_sizes = st.sidebar.checkbox("Compare file size (original vs processed)")

col1, col2 = st.columns([1,1])
orig_placeholder = col1.empty()
proc_placeholder = col2.empty()
status = st.empty()

def apply_color_conv(img, method):
    if method=="None": return img
    if method=="RGB ↔ BGR": return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if method=="RGB ↔ HSV": return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if method=="RGB ↔ YCrCb": return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if method=="Grayscale": return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def rotate_image(img, angle, scale_percent):
    (h, w) = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, scale_percent/100.0)
    return cv2.warpAffine(img, M, (w, h))

def translate_image(img, tx, ty):
    M = np.float32([[1,0,tx],[0,1,ty]])
    h,w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))

def affine_transform(img):
    h,w = img.shape[:2]
    pts1 = np.float32([[0,0],[w-1,0],[0,h-1]])
    pts2 = np.float32([[0,h*0.1],[w*0.9, h*0.05],[w*0.1,h*0.9]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (w, h))

def perspective_transform(img):
    h,w = img.shape[:2]
    pts1 = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    pts2 = np.float32([[w*0.0,h*0.05],[w*0.95,h*0.0],[w*0.9,h*0.9],[w*0.05,h*0.95]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))

def apply_filter(img, typ, k):
    if typ=="None": return img
    if typ=="Gaussian": return cv2.GaussianBlur(img, (k,k), 0)
    if typ=="Median": return cv2.medianBlur(img, k)
    if typ=="Mean": return cv2.blur(img, (k,k))
    if typ=="Sobel":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sob = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=k)
        sob = cv2.convertScaleAbs(sob)
        return cv2.cvtColor(sob, cv2.COLOR_GRAY2BGR)
    if typ=="Laplacian":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)
        return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
    return img

def apply_morph(img, op, k, iters=1):
    if op=="None": return img
    kernel = np.ones((k,k), np.uint8)
    if op=="Dilation": return cv2.dilate(img, kernel, iterations=iters)
    if op=="Erosion": return cv2.erode(img, kernel, iterations=iters)
    if op=="Opening": return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iters)
    if op=="Closing": return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return img

def histogram_equalization(img):
    if len(img.shape)==2 or img.shape[2]==1:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def contrast_stretch(img):
    in_min = np.percentile(img, 2)
    in_max = np.percentile(img, 98)
    out = (img - in_min) * (255.0/(in_max - in_min))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def edge_detect(img, method):
    if method=="None": return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method=="Canny":
        edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if method=="Sobel":
        sob = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        sob = cv2.convertScaleAbs(sob)
        return cv2.cvtColor(sob, cv2.COLOR_GRAY2BGR)
    if method=="Laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)
        return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
    return img

orig_img = None
orig_bytes = None
filename = ""
if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    orig = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if orig is not None:
        filename = uploaded.name
        orig_img = orig.copy()
        orig_bytes = file_bytes.tobytes()

if orig_img is None:
    sample = np.zeros((400,600,3), dtype=np.uint8)
    cv2.putText(sample, "Upload an Image", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
    orig_img = sample
    orig_bytes = to_bytes(orig_img, ext='.png')

orig_display = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB) if orig_img.ndim==3 else orig_img
orig_placeholder.image(orig_display, caption="Original Image", use_column_width=True)

proc = orig_img.copy()
proc = apply_color_conv(proc, col_conv)
if col_conv in ["RGB ↔ HSV","RGB ↔ YCrCb"]:
    try:
        if col_conv=="RGB ↔ HSV": proc = cv2.cvtColor(proc, cv2.COLOR_HSV2BGR)
        else: proc = cv2.cvtColor(proc, cv2.COLOR_YCrCb2BGR)
    except: pass

proc = rotate_image(proc, rot_angle, scale)
proc = translate_image(proc, tx, ty)
if do_affine: proc = affine_transform(proc)
if do_perspective: proc = perspective_transform(proc)
proc = apply_filter(proc, filter_type, kernel_size)
proc = apply_morph(proc, morph_op, kernel_size, morph_iter)

if enhance=="Histogram Equalization": proc = histogram_equalization(proc)
elif enhance=="Contrast Stretching": proc = contrast_stretch(proc)
elif enhance=="Sharpen": proc = sharpen(proc)

proc = edge_detect(proc, edge_choice)

proc_display = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB) if proc.ndim==3 else proc
proc_placeholder.image(proc_display, caption="Processed Image", use_column_width=True)

info = get_image_info(orig_img, image_bytes=orig_bytes, filename=filename)
proc_bytes = to_bytes(proc, ext='.png')
proc_info = get_image_info(proc, image_bytes=proc_bytes, filename="processed.png")
status.markdown(f"**Status** | Original: {info['width']}x{info['height']}x{info['channels']} • {info['format']} • {info['size_bytes']} bytes  | Processed: {proc_info['width']}x{proc_info['height']}x{proc_info['channels']} • {proc_info['format']} • {proc_info['size_bytes']} bytes")

if compare_sizes:
    st.write("**File size comparison:**")
    st.write(f"Original: {info['size_bytes']} bytes")
    st.write(f"Processed: {proc_info['size_bytes']} bytes")

if save_btn:
    fmt = '.png' if save_format in ['PNG','.png'] else ('.jpg' if save_format in ['.jpg','jpg'] else '.bmp')
    bts = to_bytes(proc, ext=fmt)
    st.download_button("Download processed image", data=bts, file_name=f"processed{fmt}", mime="image/png")
    st.success(f"Processed image ready for download as processed{fmt}")
