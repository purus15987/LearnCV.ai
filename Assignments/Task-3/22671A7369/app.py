"""
Streamlit Image Processing Playground
===================================

Single-file Streamlit app that demonstrates fundamental image processing operations
using OpenCV + NumPy. Users can upload an image, apply operations, and compare
Original vs Processed images side-by-side.

"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import math
import sys

st.set_page_config(page_title="Image Processing Playground", layout="wide")

# ---------------------------- Utilities ----------------------------

def pil_to_np_rgb(pil_img: Image.Image) -> np.ndarray:
    """Return RGB numpy array."""
    return np.array(pil_img.convert("RGB"))


def np_rgb_to_pil(img_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(img_rgb.astype(np.uint8))


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def ensure_odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1


def get_image_info(pil_img: Image.Image, raw_bytes: bytes = None):
    info = {}
    w, h = pil_img.size
    mode = pil_img.mode
    info['width'] = w
    info['height'] = h
    info['mode'] = mode
    info['channels'] = len(pil_img.getbands())
    info['format'] = pil_img.format if pil_img.format else 'Unknown'
    info['dpi'] = pil_img.info.get('dpi', ('Unknown',))[0] if isinstance(pil_img.info.get('dpi', None), tuple) else pil_img.info.get('dpi', 'Unknown')
    if raw_bytes is not None:
        info['file_size_bytes'] = len(raw_bytes)
    else:
        info['file_size_bytes'] = 'Unknown'
    return info


def encode_image_to_bytes(img_bgr: np.ndarray, fmt='PNG', quality=95):
    # convert BGR -> proper encoding
    fmt = fmt.upper()
    if fmt == 'JPG':
        fmt = 'JPEG'
    ext = '.' + fmt.lower()
    success, encoded = cv2.imencode(ext, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality] if fmt in ('JPEG','JPG') else [])
    if not success:
        raise RuntimeError('Failed to encode image')
    return encoded.tobytes()

# ---------------------------- Image operations ----------------------------

# Color conversions (also a manual grayscale example)

def rgb_to_gray_manual(img_rgb: np.ndarray) -> np.ndarray:
    # luminosity method
    r = img_rgb[:, :, 0].astype('float32')
    g = img_rgb[:, :, 1].astype('float32')
    b = img_rgb[:, :, 2].astype('float32')
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = np.clip(gray, 0, 255).astype('uint8')
    return gray


def equalize_hist_color(img_bgr: np.ndarray) -> np.ndarray:
    # Convert to YCrCb and equalize Y channel
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# Geometric transforms

def rotate_image(img_bgr: np.ndarray, angle_deg: float, scale: float=1.0) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    return cv2.warpAffine(img_bgr, M, (w, h))


def scale_image(img_bgr: np.ndarray, fx: float, fy: float, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(img_bgr, None, fx=fx, fy=fy, interpolation=interpolation)


def translate_image(img_bgr: np.ndarray, tx: int, ty: int) -> np.ndarray:
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = img_bgr.shape[:2]
    return cv2.warpAffine(img_bgr, M, (w, h))


def affine_transform(img_bgr: np.ndarray, src_pts, dst_pts):
    M = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))
    h, w = img_bgr.shape[:2]
    return cv2.warpAffine(img_bgr, M, (w, h))


def perspective_transform(img_bgr: np.ndarray, src_pts, dst_pts):
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    h, w = img_bgr.shape[:2]
    return cv2.warpPerspective(img_bgr, M, (w, h))

# Bitwise

def bitwise_operation(img1_bgr: np.ndarray, img2_bgr: np.ndarray, op: str):
    # ensure same size
    if img1_bgr.shape != img2_bgr.shape:
        img2_bgr = cv2.resize(img2_bgr, (img1_bgr.shape[1], img1_bgr.shape[0]))
    if op == 'AND':
        return cv2.bitwise_and(img1_bgr, img2_bgr)
    if op == 'OR':
        return cv2.bitwise_or(img1_bgr, img2_bgr)
    if op == 'XOR':
        return cv2.bitwise_xor(img1_bgr, img2_bgr)
    if op == 'NOT':
        return cv2.bitwise_not(img1_bgr)
    raise ValueError('Unknown bitwise op')

# Filters & Morphology

def apply_smoothing(img_bgr: np.ndarray, method: str, ksize: int):
    if method == 'Gaussian':
        k = ensure_odd(ksize)
        return cv2.GaussianBlur(img_bgr, (k, k), 0)
    if method == 'Median':
        k = ensure_odd(ksize)
        return cv2.medianBlur(img_bgr, k)
    if method == 'Mean':
        k = ensure_odd(ksize)
        return cv2.blur(img_bgr, (k, k))
    raise ValueError('Unknown smoothing method')


def edge_sobel(img_bgr: np.ndarray, ksize=3):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(dx**2 + dy**2)
    mag = np.clip((mag / mag.max()) * 255, 0, 255).astype('uint8')
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)


def edge_laplacian(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.clip((np.absolute(lap) / np.max(np.absolute(lap))) * 255, 0, 255).astype('uint8')
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

# Morphology helpers

def morphology(img_bgr: np.ndarray, op: str, ksize=3):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    if op == 'Dilation':
        return cv2.dilate(img_bgr, k, iterations=1)
    if op == 'Erosion':
        return cv2.erode(img_bgr, k, iterations=1)
    if op == 'Opening':
        return cv2.morphologyEx(img_bgr, cv2.MORPH_OPEN, k)
    if op == 'Closing':
        return cv2.morphologyEx(img_bgr, cv2.MORPH_CLOSE, k)
    raise ValueError('Unknown morphology op')

# Enhancement

def sharpen_image(img_bgr: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img_bgr, -1, kernel)

# ---------------------------- Session state init ----------------------------

if 'original_pil' not in st.session_state:
    st.session_state['original_pil'] = None
if 'original_bgr' not in st.session_state:
    st.session_state['original_bgr'] = None
if 'processed_bgr' not in st.session_state:
    st.session_state['processed_bgr'] = None
if 'last_action' not in st.session_state:
    st.session_state['last_action'] = 'None'

# ---------------------------- Top file area (emulated menu) ----------------------------

st.markdown("# Image Processing Playground")
with st.expander("File", expanded=True):
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        uploaded = st.file_uploader("Open → Upload an image", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'], key='uploader_top')
        if uploaded is not None:
            raw = uploaded.getvalue()
            pil = Image.open(io.BytesIO(raw))
            st.session_state['original_pil'] = pil.copy()
            rgb = pil_to_np_rgb(pil)
            st.session_state['original_bgr'] = rgb_to_bgr(rgb)
            st.session_state['processed_bgr'] = st.session_state['original_bgr'].copy()
            st.session_state['last_action'] = 'Loaded image from uploader'
    with col2:
        save_fmt = st.selectbox("Save as", ['PNG', 'JPEG', 'BMP'], index=0, key='save_format')
        save_quality = st.slider('JPEG quality (if applicable)', 10, 100, 95)
        if st.button('Save → Download processed image'):
            if st.session_state['processed_bgr'] is None:
                st.warning('No processed image to save')
            else:
                try:
                    bytes_out = encode_image_to_bytes(st.session_state['processed_bgr'], fmt=save_fmt, quality=save_quality)
                    st.download_button(label=f'Download {save_fmt}', data=bytes_out, file_name=f'processed.{save_fmt.lower()}', mime=f'image/{save_fmt.lower()}')
                except Exception as e:
                    st.error(f'Could not encode image: {e}')
    with col3:
        if st.button('Exit'):
            st.warning('Exiting app (stopping execution). Refresh to restart.')
            st.stop()

# ---------------------------- Sidebar: operations ----------------------------

st.sidebar.title('Operations')

# Second image uploader for bitwise ops
second_file = st.sidebar.file_uploader('Second image (for bitwise ops, optional)', type=['png','jpg','jpeg','bmp','tiff'], key='second_uploader')
second_bgr = None
if second_file is not None:
    raw2 = second_file.getvalue()
    pil2 = Image.open(io.BytesIO(raw2)).convert('RGB')
    second_bgr = rgb_to_bgr(np.array(pil2))

# Choose target: apply on Original or last Processed
apply_to = st.sidebar.radio('Apply operation to', ['Original', 'Processed'], index=0)

# Categories
category = st.sidebar.selectbox('Category', ['Image Info', 'Color Conversions', 'Transformations', 'Bitwise', 'Filtering & Morphology', 'Enhancement & Edge Detection', 'Compression & File Handling'])

# Parameter containers
params = {}

if category == 'Image Info':
    st.sidebar.markdown('Displays basic properties of the currently loaded image')
    if st.sidebar.button('Refresh Info'):
        st.session_state['last_action'] = 'Refreshed info'

elif category == 'Color Conversions':
    conv = st.sidebar.selectbox('Conversion', ['RGB -> BGR', 'BGR -> RGB', 'RGB -> HSV', 'HSV -> RGB', 'RGB -> YCrCb', 'YCrCb -> RGB', 'RGB -> Grayscale (OpenCV)', 'RGB -> Grayscale (Manual)'])
    params['conv'] = conv

elif category == 'Transformations':
    trans = st.sidebar.selectbox('Transform', ['Rotate', 'Scale', 'Translate', 'Affine', 'Perspective'])
    params['transform'] = trans
    if trans == 'Rotate':
        params['angle'] = st.sidebar.slider('Angle (deg)', -180, 180, 0)
        params['scale'] = st.sidebar.slider('Scale', 0.1, 3.0, 1.0)
    elif trans == 'Scale':
        params['fx'] = st.sidebar.slider('fx', 0.1, 3.0, 1.0)
        params['fy'] = st.sidebar.slider('fy', 0.1, 3.0, 1.0)
    elif trans == 'Translate':
        params['tx'] = st.sidebar.slider('tx (px)', -500, 500, 0)
        params['ty'] = st.sidebar.slider('ty (px)', -500, 500, 0)
    elif trans == 'Affine':
        st.sidebar.markdown('Provide 3 source and destination points as fractions of width/height (0..1)')
        src = [st.sidebar.text_input(f'src{i}', value=('0.0,0.0' if i==0 else ('1.0,0.0' if i==1 else '0.0,1.0'))) for i in range(3)]
        dst = [st.sidebar.text_input(f'dst{i}', value=('0.0,0.0' if i==0 else ('1.0,0.0' if i==1 else '0.0,1.0'))) for i in range(3)]
        params['src'] = src
        params['dst'] = dst
    elif trans == 'Perspective':
        st.sidebar.markdown('Provide 4 source and destination points as fractions of width/height (0..1)')
        src = [st.sidebar.text_input(f'psrc{i}', value=('0.0,0.0' if i==0 else ('1.0,0.0' if i==1 else ('1.0,1.0' if i==2 else '0.0,1.0')))) for i in range(4)]
        dst = [st.sidebar.text_input(f'pdst{i}', value=('0.0,0.0' if i==0 else ('1.0,0.0' if i==1 else ('1.0,1.0' if i==2 else '0.0,1.0')))) for i in range(4)]
        params['src'] = src
        params['dst'] = dst

elif category == 'Bitwise':
    op = st.sidebar.selectbox('Bitwise op', ['AND', 'OR', 'XOR', 'NOT'])
    params['bitwise_op'] = op
    st.sidebar.markdown('When using NOT, the second image is ignored.')

elif category == 'Filtering & Morphology':
    filt = st.sidebar.selectbox('Filter / Morph', ['Gaussian Blur', 'Median Blur', 'Mean Blur', 'Sobel', 'Laplacian', 'Dilation', 'Erosion', 'Opening', 'Closing'])
    params['filter'] = filt
    if 'Blur' in filt:
        params['ksize'] = st.sidebar.slider('Kernel size', 1, 31, 3)
    elif filt in ['Dilation','Erosion','Opening','Closing']:
        params['m_ksize'] = st.sidebar.slider('Structuring element size', 1, 31, 3)
    elif filt in ['Sobel']:
        params['sobel_k'] = st.sidebar.selectbox('Sobel ksize', [1,3,5,7], index=1)

elif category == 'Enhancement & Edge Detection':
    enh = st.sidebar.selectbox('Operation', ['Histogram Equalization', 'Sharpen', 'Canny Edge', 'Sobel Edge', 'Laplacian Edge'])
    params['enh'] = enh
    if enh == 'Canny Edge':
        params['canny_t1'] = st.sidebar.slider('Canny threshold1', 0, 500, 100)
        params['canny_t2'] = st.sidebar.slider('Canny threshold2', 0, 500, 200)

elif category == 'Compression & File Handling':
    st.sidebar.markdown('Save processed image and compare sizes')
    comp_fmt = st.sidebar.selectbox('Format for comparison', ['PNG','JPEG','BMP'])
    comp_quality = st.sidebar.slider('JPEG quality', 10, 100, 95)
    params['comp_fmt'] = comp_fmt
    params['comp_quality'] = comp_quality

# Apply button
apply_clicked = st.sidebar.button('Apply Operation')
reset_clicked = st.sidebar.button('Reset to original')

# ---------------------------- Processing logic ----------------------------

def get_active_img():
    src = st.session_state['original_bgr'] if apply_to == 'Original' else st.session_state['processed_bgr']
    return None if src is None else src.copy()

if reset_clicked:
    if st.session_state['original_bgr'] is not None:
        st.session_state['processed_bgr'] = st.session_state['original_bgr'].copy()
        st.session_state['last_action'] = 'Reset to original'

if apply_clicked:
    base = get_active_img()
    if base is None:
        st.warning('No image loaded. Use File → Open to upload an image.')
    else:
        try:
            if category == 'Image Info':
                st.session_state['last_action'] = 'Displayed info'

            elif category == 'Color Conversions':
                conv = params['conv']
                if conv == 'RGB -> BGR':
                    out = cv2.cvtColor(bgr_to_rgb(base), cv2.COLOR_RGB2BGR)
                elif conv == 'BGR -> RGB':
                    out = bgr_to_rgb(base)
                    out = rgb_to_bgr(out)
                elif conv == 'RGB -> HSV':
                    out = cv2.cvtColor(bgr_to_rgb(base), cv2.COLOR_RGB2HSV)
                    out = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
                    out = rgb_to_bgr(out)
                elif conv == 'HSV -> RGB':
                    # we assume the base is in BGR; convert to HSV then back
                    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
                    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                elif conv == 'RGB -> YCrCb':
                    out = cv2.cvtColor(bgr_to_rgb(base), cv2.COLOR_RGB2YCrCb)
                    out = cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)
                    out = rgb_to_bgr(out)
                elif conv == 'YCrCb -> RGB':
                    ycb = cv2.cvtColor(base, cv2.COLOR_BGR2YCrCb)
                    out = cv2.cvtColor(ycb, cv2.COLOR_YCrCb2BGR)
                elif conv == 'RGB -> Grayscale (OpenCV)':
                    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
                    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                elif conv == 'RGB -> Grayscale (Manual)':
                    gray = rgb_to_gray_manual(bgr_to_rgb(base))
                    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                st.session_state['processed_bgr'] = out
                st.session_state['last_action'] = f'Color conversion: {conv}'

            elif category == 'Transformations':
                t = params['transform']
                if t == 'Rotate':
                    out = rotate_image(base, params['angle'], params['scale'])
                elif t == 'Scale':
                    out = scale_image(base, params['fx'], params['fy'])
                elif t == 'Translate':
                    out = translate_image(base, params['tx'], params['ty'])
                elif t == 'Affine':
                    h, w = base.shape[:2]
                    def parse_point(s):
                        x, y = s.split(',')
                        return (float(x) * w, float(y) * h)
                    src_pts = [parse_point(s) for s in params['src']]
                    dst_pts = [parse_point(s) for s in params['dst']]
                    out = affine_transform(base, src_pts, dst_pts)
                elif t == 'Perspective':
                    h, w = base.shape[:2]
                    def parse_point(s):
                        x, y = s.split(',')
                        return (float(x) * w, float(y) * h)
                    src_pts = [parse_point(s) for s in params['src']]
                    dst_pts = [parse_point(s) for s in params['dst']]
                    out = perspective_transform(base, src_pts, dst_pts)
                st.session_state['processed_bgr'] = out
                st.session_state['last_action'] = f'Transform: {t}'

            elif category == 'Bitwise':
                op = params['bitwise_op']
                if op == 'NOT':
                    out = bitwise_operation(base, base, 'NOT')
                else:
                    if second_bgr is None:
                        st.warning('Provide a second image in the sidebar for AND/OR/XOR ops. Using inverted base as second image.')
                        out = bitwise_operation(base, cv2.bitwise_not(base), op)
                    else:
                        out = bitwise_operation(base, second_bgr, op)
                st.session_state['processed_bgr'] = out
                st.session_state['last_action'] = f'Bitwise: {op}'

            elif category == 'Filtering & Morphology':
                f = params['filter']
                if f in ['Gaussian Blur','Median Blur','Mean Blur']:
                    k = params['ksize']
                    method = 'Gaussian' if 'Gaussian' in f else ('Median' if 'Median' in f else 'Mean')
                    out = apply_smoothing(base, method, k)
                elif f in ['Sobel']:
                    k = params['sobel_k']
                    out = edge_sobel(base, ksize=k)
                elif f in ['Laplacian']:
                    out = edge_laplacian(base)
                elif f in ['Dilation','Erosion','Opening','Closing']:
                    k = params['m_ksize']
                    out = morphology(base, f if f != 'Mean Blur' else 'Dilation', k)
                st.session_state['processed_bgr'] = out
                st.session_state['last_action'] = f'Filter/Morph: {f}'

            elif category == 'Enhancement & Edge Detection':
                e = params['enh']
                if e == 'Histogram Equalization':
                    out = equalize_hist_color(base)
                elif e == 'Sharpen':
                    out = sharpen_image(base)
                elif e == 'Canny Edge':
                    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
                    t1, t2 = params['canny_t1'], params['canny_t2']
                    can = cv2.Canny(gray, t1, t2)
                    out = cv2.cvtColor(can, cv2.COLOR_GRAY2BGR)
                elif e == 'Sobel Edge':
                    out = edge_sobel(base)
                elif e == 'Laplacian Edge':
                    out = edge_laplacian(base)
                st.session_state['processed_bgr'] = out
                st.session_state['last_action'] = f'Enhancement/Edge: {e}'

            elif category == 'Compression & File Handling':
                fmt = params['comp_fmt']
                qual = params['comp_quality']
                bytes_out = encode_image_to_bytes(st.session_state['processed_bgr'], fmt=fmt, quality=qual)
                st.sidebar.markdown(f'*Encoded {fmt} size: {len(bytes_out)} bytes*')
                st.session_state['last_action'] = f'Compression to {fmt} at q={qual}'

            else:
                st.warning('Category not handled yet')

        except Exception as e:
            st.error(f'Error while applying operation: {e}')

# ---------------------------- Main display area ----------------------------

orig_col, proc_col = st.columns(2)
with orig_col:
    st.subheader('Original Image')
    if st.session_state['original_bgr'] is not None:
        st.image(bgr_to_rgb(st.session_state['original_bgr']), use_column_width=True)
    else:
        st.info('No image loaded. Use File → Open to upload an image.')

with proc_col:
    st.subheader('Processed Image')
    if st.session_state['processed_bgr'] is not None:
        st.image(bgr_to_rgb(st.session_state['processed_bgr']), use_column_width=True)
    else:
        st.info('Processed image will appear here.')

# ---------------------------- Bottom status bar ----------------------------

st.markdown('---')
status_col1, status_col2, status_col3 = st.columns([1,1,2])
with status_col1:
    st.markdown('**Dimensions (H x W x C)**')
    if st.session_state['processed_bgr'] is not None:
        h, w, c = st.session_state['processed_bgr'].shape
        st.write(f'{h} x {w} x {c}')
    elif st.session_state['original_bgr'] is not None:
        h, w, c = st.session_state['original_bgr'].shape
        st.write(f'{h} x {w} x {c}')
    else:
        st.write('N/A')

with status_col2:
    st.markdown('**File Format & DPI**')
    if st.session_state['original_pil'] is not None:
        info = get_image_info(st.session_state['original_pil'])
        st.write(f"Format: {info['format']}")
        st.write(f"DPI: {info['dpi']}")
    else:
        st.write('N/A')

with status_col3:
    st.markdown('**File size**')
    uploaded_obj = st.session_state.get('original_pil', None)
    # try to compute size
    if st.session_state.get('original_bgr', None) is not None:
        try:
            raw_bytes = None
            if 'uploader_top' in st.session_state and st.session_state['original_pil'] is not None:
                # try to use the uploader object (can't always access raw bytes later), so safe to show Unknown
                st.write('Original file size may be shown in upload dialogue (or Unknown)')
            else:
                st.write('Unknown')
        except Exception:
            st.write('Unknown')
    else:
        st.write('N/A')

st.markdown(f"**Last action:** {st.session_state.get('last_action','None')}")

# ---------------------------- Footer ----------------------------

st.caption('Tip: use the sidebar to switch categories and configure parameters. Click "Apply Operation" to update the processed image. Use "Reset to original" to revert.')

# End of file
