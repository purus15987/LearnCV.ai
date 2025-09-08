import streamlit as st 
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os

st.set_page_config(page_title="Image Processing Toolkit", layout="wide")

# --- Utility functions -----------------------------------------------------

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    return np.array(image)


def to_pil(img_arr):
    if img_arr.dtype != np.uint8:
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(img_arr)


def get_image_info(img_arr, uploaded_file=None):
    h, w = img_arr.shape[:2]
    c = 1 if img_arr.ndim == 2 else img_arr.shape[2]
    filesize = None
    fmt = None
    dpi = (72, 72)
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0, os.SEEK_END)
            filesize = uploaded_file.tell()
            uploaded_file.seek(0)
        except Exception:
            filesize = None
        fmt = getattr(uploaded_file, 'type', None)
    return {
        'Height': h,
        'Width': w,
        'Channels': c,
        'File Size (bytes)': filesize,
        'DPI': dpi,
        'Format': fmt,
    }


# Color conversion helpers

def convert_color(img, mode):
    # img is RGB numpy array
    if mode == 'BGR':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if mode == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if mode == 'YCbCr':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if mode == 'GRAY':
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


# Transformations

def rotate_image(img, angle, center=None, scale=1.0):
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (w, h))


def scale_image(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)


def translate_image(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def affine_transform(img, src_pts, dst_pts, dsize=None):
    M = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))
    if dsize is None:
        h, w = img.shape[:2]
        dsize = (w, h)
    return cv2.warpAffine(img, M, dsize)


def perspective_transform(img, src_pts, dst_pts, dsize=None):
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    if dsize is None:
        h, w = img.shape[:2]
        dsize = (w, h)
    return cv2.warpPerspective(img, M, dsize)


# Filtering & morphology

def apply_filter(img, method, ksize=3):
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if method == 'median':
        return cv2.medianBlur(img, ksize)
    if method == 'mean':
        return cv2.blur(img, (ksize, ksize))
    return img


def morphological(img, op, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    if op == 'dilate':
        return cv2.dilate(img, kernel)
    if op == 'erode':
        return cv2.erode(img, kernel)
    if op == 'open':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if op == 'close':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


# Enhancement

def histogram_equalization(img):
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def contrast_stretch(img, in_min=0, in_max=255):
    out = (img - in_min) * (255.0 / (in_max - in_min))
    return np.clip(out, 0, 255).astype(np.uint8)


def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


# Edge detectors

def sobel_edge(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.uint8(np.clip(mag / mag.max() * 255, 0, 255))
    return mag


def laplacian_edge(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.uint8(np.clip(np.abs(lap) / np.abs(lap).max() * 255, 0, 255))
    return lap


def canny_edge(img, t1, t2):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(img, t1, t2)


# Compression helper

def save_image_bytes(img_arr, fmt='PNG', quality=95):
    pil = to_pil(img_arr)
    buf = BytesIO()
    save_kwargs = {}
    if fmt.upper() == 'JPEG':
        save_kwargs['format'] = 'JPEG'
        save_kwargs['quality'] = quality
    else:
        save_kwargs['format'] = fmt
    pil.save(buf, **save_kwargs)
    buf.seek(0)
    return buf


# --- Streamlit UI ---------------------------------------------------------

st.title('📸 Image Processing & Analysis Toolkit')

# Top menu (simple emulation)
menu_col1, menu_col2, menu_col3 = st.columns([1, 6, 1])
with menu_col1:
    if st.button('Open'):
        st.sidebar.info('Use the Upload control in the sidebar to open an image.')
with menu_col3:
    if st.button('Exit'):
        st.stop()

# Sidebar
st.sidebar.header('📂 File')
uploaded_file = st.sidebar.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg', 'bmp'])

st.sidebar.markdown('---')

st.sidebar.header('🛠 Operations')
show_info = st.sidebar.checkbox('Show Image Info', value=True)

color_ops = st.sidebar.selectbox('Color Conversion', ['None', 'BGR', 'HSV', 'YCbCr', 'GRAY'])

st.sidebar.markdown('---')
transform = st.sidebar.selectbox('Transformation', ['None', 'Rotate', 'Scale', 'Translate', 'Affine', 'Perspective'])

filter_ops = st.sidebar.selectbox('Filtering', ['None', 'Gaussian', 'Median', 'Mean'])
ksize = st.sidebar.slider('Kernel size (odd)', 1, 31, 3, step=2)

morph_ops = st.sidebar.selectbox('Morphology', ['None', 'Dilate', 'Erode', 'Open', 'Close'])
morph_ks = st.sidebar.slider('Morph kernel', 1, 31, 3, step=2)

st.sidebar.markdown('---')

enhance_ops = st.sidebar.selectbox('Enhancement', ['None', 'Histogram Equalization', 'Contrast Stretch', 'Sharpen'])

edge_ops = st.sidebar.selectbox('Edge Detection', ['None', 'Sobel', 'Laplacian', 'Canny'])
can_t1 = st.sidebar.slider('Canny t1', 0, 500, 100)
can_t2 = st.sidebar.slider('Canny t2', 0, 500, 200)

st.sidebar.markdown('---')

compression_fmt = st.sidebar.selectbox('Save Format', ['PNG', 'JPEG', 'BMP'])
jpeg_quality = st.sidebar.slider('JPEG Quality', 10, 100, 95)

st.sidebar.markdown('---')

# Bonus
st.sidebar.header('🎁 Bonus')
show_split = st.sidebar.checkbox('Comparison split-screen (half/half)')
realtime = st.sidebar.checkbox('Webcam (real-time mode)')

# Placeholders
col1, col2 = st.columns(2)
orig_placeholder = col1.empty()
proc_placeholder = col2.empty()

status = st.empty()

# If webcam mode
if realtime:
    run = st.checkbox('Run Webcam')
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error('Cannot open webcam')
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                proc = frame.copy()
                if edge_ops == 'Canny':
                    proc = canny_edge(proc, can_t1, can_t2)
                proc_placeholder.image(proc, use_column_width=True)
                orig_placeholder.image(frame, use_column_width=True)
                if st.session_state.get('stop_webcam'):
                    break
            cap.release()

# Load static image
if uploaded_file is not None:
    img = load_image(uploaded_file)
    orig = img.copy()

    # show info
    info = get_image_info(img, uploaded_file)

    # Color conversion
    if color_ops != 'None':
        img = convert_color(img, color_ops)

    # Transformations
    if transform == 'Rotate':
        angle = st.sidebar.slider('Angle', -180, 180, 0)
        img = rotate_image(img, angle)
    elif transform == 'Scale':
        fx = st.sidebar.slider('Scale X', 0.1, 3.0, 1.0)
        fy = st.sidebar.slider('Scale Y', 0.1, 3.0, 1.0)
        img = scale_image(img, fx, fy)
    elif transform == 'Translate':
        tx = st.sidebar.slider('Translate X', -200, 200, 0)
        ty = st.sidebar.slider('Translate Y', -200, 200, 0)
        img = translate_image(img, tx, ty)
    elif transform == 'Affine':
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
        dst = np.float32([[0, 0], [w - 1, 0], [int(0.2 * w), int(0.4 * h)]])
        img = affine_transform(img, src, dst)
    elif transform == 'Perspective':
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([[0, int(0.1 * h)], [w - 1, 0], [int(0.9 * w), h - 1], [int(0.1 * w), int(0.9 * h)]])
        img = perspective_transform(img, src, dst)

    # Filtering
    if filter_ops != 'None':
        img = apply_filter(img, filter_ops.lower(), ksize)

    # Morphology
    if morph_ops != 'None':
        op_map = {'Dilate': 'dilate', 'Erode': 'erode', 'Open': 'open', 'Close': 'close'}
        img = morphological(img, op_map[morph_ops], morph_ks)

    # Enhancement
    if enhance_ops == 'Histogram Equalization':
        img = histogram_equalization(img)
    elif enhance_ops == 'Contrast Stretch':
        img = contrast_stretch(img)
    elif enhance_ops == 'Sharpen':
        img = sharpen(img)

    # Edge detection
    if edge_ops == 'Sobel':
        img = sobel_edge(img)
    elif edge_ops == 'Laplacian':
        img = laplacian_edge(img)
    elif edge_ops == 'Canny':
        img = canny_edge(img, can_t1, can_t2)

    # Display
    if show_split:
        # make half-half composite
        h = max(orig.shape[0], img.shape[0])
        w = orig.shape[1] + (img.shape[1] if img.ndim == 3 else orig.shape[1])
        # Create canvas
        left = orig.copy()
        right = img.copy()
        try:
            composite = np.hstack((left, right if right.ndim == 3 else cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)))
        except Exception:
            composite = left
        proc_placeholder.image(composite, use_column_width=True)
        orig_placeholder.image(orig, caption='Original')
    else:
        orig_placeholder.image(orig, caption='Original')
        proc_placeholder.image(img, caption='Processed')

    # Status bar
    status.markdown(f"**Dimensions:** {info['Height']} x {info['Width']} | **Channels:** {info['Channels']} | **Format:** {info['Format']} | **File size (bytes):** {info['File Size (bytes)']}")

    # Save processed image
    st.markdown('---')
    st.subheader('Save / Export')
    if st.button('Save Processed Image'):
        buf = save_image_bytes(img, fmt=compression_fmt, quality=jpeg_quality)
        st.download_button('Download', data=buf, file_name=f'processed.{compression_fmt.lower()}', mime=f'image/{compression_fmt.lower()}')

    # Compare sizes
    if st.button('Compare file sizes (Original vs Processed)'):
        orig_buf = save_image_bytes(orig, fmt='PNG')
        proc_buf = save_image_bytes(img, fmt=compression_fmt, quality=jpeg_quality)
        orig_size = len(orig_buf.getvalue())
        proc_size = len(proc_buf.getvalue())
        st.info(f'Original (PNG): {orig_size} bytes — Processed ({compression_fmt}): {proc_size} bytes')
else:
    st.info('Upload an image to begin — supported types: png, jpg, jpeg, bmp')


# Footer / help
st.markdown('---')
st.caption('Built with Streamlit + OpenCV — Assignment 3: Image Processing Toolkit')
