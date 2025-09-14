# ImageProcessingToolkit_app.py
# Streamlit + OpenCV + NumPy GUI for Image Processing Toolkit
# Assignment 3 — Image Processing & Analysis Toolkit

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

st.set_page_config(page_title="Image Processing Toolkit", layout="wide")

# ----------------------- Utility Functions -----------------------

def to_pil(img: np.ndarray):
    """Convert an OpenCV BGR or grayscale image to PIL RGB for Streamlit display."""
    if img is None:
        return None
    if len(img.shape) == 2:
        return Image.fromarray(img)
    # assume BGR -> convert to RGB
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def read_image(file) -> (np.ndarray, dict):
    # read bytes and convert to OpenCV image (BGR)
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    props = {}
    if img is None:
        return None, props
    props['shape'] = img.shape
    try:
        # approximate DPI if present in PIL tags
        pil = Image.open(io.BytesIO(file_bytes))
        props['format'] = pil.format
        info = pil.info
        props['dpi'] = info.get('dpi', (72,72))
    except Exception:
        props['format'] = 'UNKNOWN'
        props['dpi'] = (72,72)
    # restore file pointer for possible re-use
    file.seek(0)
    return img, props


def show_status(col, img, props):
    if img is None:
        col.write("No image loaded")
        return
    h, w = img.shape[:2]
    c = 1 if len(img.shape) == 2 else img.shape[2]
    col.markdown(f"**Dimensions:** {h} x {w} (H x W)")
    col.markdown(f"**Channels:** {c}")
    fmt = props.get('format', 'UNKNOWN')
    col.markdown(f"**Format:** {fmt}")
    dpi = props.get('dpi', (72,72))
    col.markdown(f"**DPI/PPI:** {dpi[0]} x {dpi[1]}")


def save_image(img: np.ndarray, out_path: str, fmt='PNG', quality=95):
    # convert BGR to RGB for PIL
    if img is None:
        return False
    pil = to_pil(img)
    try:
        if fmt.upper() in ['JPG','JPEG']:
            pil.save(out_path, format='JPEG', quality=quality)
        else:
            pil.save(out_path, format=fmt)
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False


# ----------------------- Image Processing Ops -----------------------

def convert_color(img, mode):
    if img is None:
        return None
    if mode == 'BGR->RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if mode == 'BGR->GRAY':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == 'BGR->HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if mode == 'BGR->YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img


def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def scale_image(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)


def translate_image(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def affine_transform(img, pts_src, pts_dst):
    M = cv2.getAffineTransform(np.float32(pts_src), np.float32(pts_dst))
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def perspective_transform(img, pts_src, pts_dst):
    M = cv2.getPerspectiveTransform(np.float32(pts_src), np.float32(pts_dst))
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h))


def apply_filter(img, typ, ksize=3):
    if img is None:
        return None
    if typ == 'Gaussian':
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if typ == 'Median':
        return cv2.medianBlur(img, ksize)
    if typ == 'Mean':
        return cv2.blur(img, (ksize, ksize))
    return img


def morphology(img, op, ksize=3):
    kernel = np.ones((ksize,ksize), np.uint8)
    if op == 'Erode':
        return cv2.erode(img, kernel, iterations=1)
    if op == 'Dilate':
        return cv2.dilate(img, kernel, iterations=1)
    if op == 'Open':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if op == 'Close':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def edge_detect(img, method='Canny', thresh1=100, thresh2=200):
    gray = img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == 'Canny':
        return cv2.Canny(gray, thresh1, thresh2)
    if method == 'Sobel':
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sob = np.hypot(sx, sy)
        sob = np.uint8(np.clip(sob / np.max(sob) * 255, 0, 255))
        return sob
    if method == 'Laplacian':
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.uint8(np.clip(np.absolute(lap), 0, 255))
        return lap
    return gray


def histogram_equalize(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    # convert to YCrCb and equalize Y channel
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)


def bitwise_op(img1, img2, op):
    if img1 is None or img2 is None:
        return None
    # ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    if op == 'AND':
        return cv2.bitwise_and(img1, img2)
    if op == 'OR':
        return cv2.bitwise_or(img1, img2)
    if op == 'XOR':
        return cv2.bitwise_xor(img1, img2)
    if op == 'NOT (img1)':
        return cv2.bitwise_not(img1)
    return None


# ----------------------- Streamlit Layout -----------------------

st.title("🖼 Image Processing & Analysis Toolkit — Streamlit + OpenCV")

# Top menu (light emulation)
menu_col1, menu_col2 = st.columns([6,1])
with menu_col2:
    if st.button('Exit'):
        st.warning('To close the app, stop the Streamlit process in your terminal')

# Sidebar - file operations & controls
st.sidebar.header('📂 File')
uploaded = st.sidebar.file_uploader('Open → Upload an image', type=['png','jpg','jpeg','bmp','tif','tiff'])

save_fmt = st.sidebar.selectbox('Save format', ['PNG','JPG','BMP'])
quality = st.sidebar.slider('JPG Quality', 50, 100, 95) if save_fmt in ['JPG','JPEG'] else None

st.sidebar.markdown('---')

# Operations categories
st.sidebar.header('🔧 Operations')
ops_cat = st.sidebar.selectbox('Category', ['Image Info','Color Conversions','Transformations','Filtering & Morphology','Enhancement','Edge Detection','Bitwise Ops','Compression'])

# interactive parameters
angle = st.sidebar.slider('Rotation angle', -180, 180, 0)
scale = st.sidebar.slider('Scaling factor', 10, 300, 100)
scale = scale / 100.0
trans_x = st.sidebar.slider('Translate X', -200, 200, 0)
trans_y = st.sidebar.slider('Translate Y', -200, 200, 0)
ksize = st.sidebar.slider('Kernel size (odd)', 1, 31, 3, step=2)
edge_thresh1 = st.sidebar.slider('Edge thresh1', 0, 500, 100)
edge_thresh2 = st.sidebar.slider('Edge thresh2', 0, 500, 200)

# placeholders for additional param inputs

# load image
orig_img = None
img_props = {}
if uploaded is not None:
    orig_img, img_props = read_image(uploaded)

# Persist processed image in session state
if 'processed' not in st.session_state:
    st.session_state['processed'] = None

# Main display area
col1, col2 = st.columns(2)
with col1:
    st.subheader('Original Image')
    if orig_img is None:
        st.info('Upload an image from the sidebar to get started')
    else:
        st.image(to_pil(orig_img), use_column_width=True)
with col2:
    st.subheader('Processed Image')
    # perform selected operation
    processed = None
    if ops_cat == 'Image Info':
        processed = orig_img
    elif ops_cat == 'Color Conversions':
        conv = st.sidebar.selectbox('Convert', ['BGR->RGB','BGR->GRAY','BGR->HSV','BGR->YCrCb'])
        processed = convert_color(orig_img, conv)
        # some conversions return single channel - keep as is
    elif ops_cat == 'Transformations':
        t = st.sidebar.selectbox('Transform', ['Rotate','Scale','Translate','Affine','Perspective'])
        if t == 'Rotate':
            processed = rotate_image(orig_img, angle) if orig_img is not None else None
        elif t == 'Scale':
            processed = scale_image(orig_img, scale, scale) if orig_img is not None else None
        elif t == 'Translate':
            processed = translate_image(orig_img, trans_x, trans_y) if orig_img is not None else None
        elif t == 'Affine':
            st.sidebar.write('Affine uses example source/dest points (drag not supported)')
            h,w = (orig_img.shape[0], orig_img.shape[1]) if orig_img is not None else (0,0)
            pts_src = [[0,0],[w-1,0],[0,h-1]]
            pts_dst = [[0,0],[int(0.8*(w-1)), int(0.1*h)],[int(0.1*w), int(0.9*h)]]
            processed = affine_transform(orig_img, pts_src, pts_dst) if orig_img is not None else None
        else:
            st.sidebar.write('Perspective uses example corners')
            h,w = (orig_img.shape[0], orig_img.shape[1]) if orig_img is not None else (0,0)
            pts_src = [[0,0],[w-1,0],[w-1,h-1],[0,h-1]]
            pts_dst = [[int(0.0*w),int(0.05*h)],[int(0.95*w),int(0.0*h)],[int(0.9*w),int(0.95*h)],[int(0.05*w),int(0.9*h)]]
            processed = perspective_transform(orig_img, pts_src, pts_dst) if orig_img is not None else None
    elif ops_cat == 'Filtering & Morphology':
        f = st.sidebar.selectbox('Filter / Morph', ['Gaussian','Median','Mean','Erode','Dilate','Open','Close','Sobel','Laplacian'])
        if f in ['Gaussian','Median','Mean']:
            processed = apply_filter(orig_img, f, ksize)
        elif f in ['Erode','Dilate','Open','Close']:
            processed = morphology(orig_img, f, ksize)
        elif f == 'Sobel':
            processed = edge_detect(orig_img, method='Sobel')
        elif f == 'Laplacian':
            processed = edge_detect(orig_img, method='Laplacian')
    elif ops_cat == 'Enhancement':
        e = st.sidebar.selectbox('Enhancement', ['Histogram Equalization','Sharpen'])
        if e == 'Histogram Equalization':
            processed = histogram_equalize(orig_img)
        else:
            processed = sharpen(orig_img)
    elif ops_cat == 'Edge Detection':
        m = st.sidebar.selectbox('Method', ['Canny','Sobel','Laplacian'])
        processed = edge_detect(orig_img, method=m, thresh1=edge_thresh1, thresh2=edge_thresh2)
    elif ops_cat == 'Bitwise Ops':
        op = st.sidebar.selectbox('Operation', ['AND','OR','XOR','NOT (img1)'])
        st.sidebar.markdown('Upload a second image to combine (optional)')
        second = st.sidebar.file_uploader('Second image', type=['png','jpg','jpeg','bmp','tif','tiff'], key='second')
        img2 = None
        if second is not None:
            img2, _ = read_image(second)
        processed = bitwise_op(orig_img, img2, op)
    elif ops_cat == 'Compression':
        st.sidebar.write('Choose save format & compare file sizes')
        processed = orig_img

    # Show processed image
    if processed is None:
        st.info('No processed output to show — check image & selected operation')
    else:
        # if single channel, show as PIL directly
        st.image(to_pil(processed), use_column_width=True)
    # store
    st.session_state['processed'] = processed

# Bottom status bar
st.markdown('---')
status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    st.markdown('**Original Image Info**')
    show_status(st, orig_img, img_props)
with status_col2:
    st.markdown('**Processed Image Info**')
    show_status(st, st.session_state.get('processed', None), {})
with status_col3:
    if uploaded is not None:
        uploaded_size = uploaded.size
        st.markdown(f"**Original File Size:** {uploaded_size} bytes")
    if st.session_state.get('processed', None) is not None:
        st.download_button('Save Processed Image', data=io.BytesIO(cv2.imencode('.png', st.session_state['processed'])[1].tobytes()), file_name=f'processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png')

# Footer / tips
st.markdown('---')
st.caption('Tips: Use sliders on the left to adjust parameters. For webcam mode and advanced features, extend the app with `cv2.VideoCapture(0)` and frame processing loops.')

# End of file
