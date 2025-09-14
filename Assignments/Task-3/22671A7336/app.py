import streamlit as st

import numpy as np

import cv2

from PIL import Image

import io

import os

from datetime import datetime



st.set_page_config(page_title="Image Processing Toolkit", layout="wide")



# -------------------- Helpers --------------------



def to_bytes(img, fmt='PNG'):

  is_success, buffer = cv2.imencode('.' + fmt.lower(), img)

  if not is_success:

    return None

  return buffer.tobytes()





def get_image_info(img_bytes):

  # img_bytes: bytes

  try:

    img = Image.open(io.BytesIO(img_bytes))

    width, height = img.size

    mode = img.mode

    fmt = img.format

    # DPI if present

    dpi = img.info.get('dpi', (0, 0))

    file_size = len(img_bytes)

    channels = len(img.getbands())

    return {

      'width': width,

      'height': height,

      'mode': mode,

      'format': fmt,

      'dpi': dpi,

      'size_bytes': file_size,

      'channels': channels

    }

  except Exception as e:

    return None





def pil_to_cv2(pil_img):

  img = np.array(pil_img)

  if img.ndim == 2:

    return img

  # convert RGB to BGR

  return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)





def cv2_to_pil(cv_img):

  if cv_img.ndim == 2:

    return Image.fromarray(cv_img)

  img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

  return Image.fromarray(img)





def ensure_color(img):

  # Ensure image is BGR 3-channel numpy array

  if img is None:

    return None

  if isinstance(img, Image.Image):

    img = pil_to_cv2(img)

  if img.ndim == 2:

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

  if img.shape[2] == 4:

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

  return img



# -------------------- Conversions (manual & OpenCV) --------------------



def bgr_to_rgb_manual(bgr):

  # simple channel swap

  return bgr[:, :, ::-1]





def bgr_to_gray_manual(bgr):

  # using luminosity method Y = 0.299 R + 0.587 G + 0.114 B

  B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

  gray = 0.114 * B + 0.587 * G + 0.299 * R

  return gray.astype(np.uint8)





def bgr_to_ycbcr_manual(bgr):

  # formula from standard (approx)

  B = bgr[:, :, 0].astype(np.float32)

  G = bgr[:, :, 1].astype(np.float32)

  R = bgr[:, :, 2].astype(np.float32)

  Y = 0.299*R + 0.587*G + 0.114*B

  Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B

  Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B

  ycbcr = np.stack([Y, Cb, Cr], axis=2)

  return np.clip(ycbcr, 0, 255).astype(np.uint8)



# -------------------- Transformations --------------------



def rotate_image(img, angle, center=None, scale=1.0):

  (h, w) = img.shape[:2]

  if center is None:

    center = (w // 2, h // 2)

  M = cv2.getRotationMatrix2D(center, angle, scale)

  rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

  return rotated





def scale_image(img, fx, fy):

  return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)





def translate_image(img, tx, ty):

  M = np.float32([[1, 0, tx], [0, 1, ty]])

  (h, w) = img.shape[:2]

  return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)





def affine_transform(img, pts_src, pts_dst):

  M = cv2.getAffineTransform(np.float32(pts_src), np.float32(pts_dst))

  (h, w) = img.shape[:2]

  return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)





def perspective_transform(img, src_pts, dst_pts):

  M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))

  (h, w) = img.shape[:2]

  return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)



# -------------------- Filters & Morphology --------------------



def apply_smoothing(img, method='gaussian', ksize=3):

  if ksize % 2 == 0:

    ksize += 1

  if method == 'gaussian':

    return cv2.GaussianBlur(img, (ksize, ksize), 0)

  elif method == 'median':

    return cv2.medianBlur(img, ksize)

  elif method == 'mean':

    return cv2.blur(img, (ksize, ksize))

  else:

    return img





def apply_edge_filter(img, method='sobel'):

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  if method == 'sobel':

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    sob = np.hypot(sx, sy)

    sob = np.uint8(np.clip(sob / sob.max() * 255, 0, 255))

    return cv2.cvtColor(sob, cv2.COLOR_GRAY2BGR)

  elif method == 'laplacian':

    lap = cv2.Laplacian(gray, cv2.CV_64F)

    lap = np.uint8(np.clip(np.abs(lap) / np.abs(lap).max() * 255, 0, 255))

    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

  elif method == 'canny':

    can = cv2.Canny(gray, 100, 200)

    return cv2.cvtColor(can, cv2.COLOR_GRAY2BGR)

  else:

    return img





def apply_morphology(img, op='dilate', kernel_size=3, iterations=1):

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  if op == 'dilate':

    res = cv2.dilate(gray, kernel, iterations=iterations)

  elif op == 'erode':

    res = cv2.erode(gray, kernel, iterations=iterations)

  elif op == 'open':

    res = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

  elif op == 'close':

    res = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

  else:

    res = gray

  return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)



# -------------------- Enhancement --------------------



def histogram_equalization(img):

  if img.ndim == 3 and img.shape[2] == 3:

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    y, cr, cb = cv2.split(ycrcb)

    y_eq = cv2.equalizeHist(y)

    ycrcb_eq = cv2.merge([y_eq, cr, cb])

    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

  else:

    eq = cv2.equalizeHist(img)

    return eq





def sharpen_image(img):

  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

  return cv2.filter2D(img, -1, kernel)



# -------------------- Bitwise Ops --------------------



def bitwise_ops(img1, img2, op='and'):

  img1_g = ensure_color(img1)

  img2_g = ensure_color(img2)

  img2_resized = cv2.resize(img2_g, (img1_g.shape[1], img1_g.shape[0]))

  if op == 'and':

    return cv2.bitwise_and(img1_g, img2_resized)

  if op == 'or':

    return cv2.bitwise_or(img1_g, img2_resized)

  if op == 'xor':

    return cv2.bitwise_xor(img1_g, img2_resized)

  if op == 'not':

    return cv2.bitwise_not(img1_g)

  return img1_g



# -------------------- Compression Helpers --------------------



def get_saved_size_and_bytes(img_cv, fmt='PNG', quality=95):

  ext = fmt.upper()

  is_success, buffer = cv2.imencode('.' + fmt.lower(), img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), quality]) if fmt.upper()=='JPG' or fmt.upper()=='JPEG' else cv2.imencode('.' + fmt.lower(), img_cv)

  if not is_success:

    return None

  b = buffer.tobytes()

  return len(b), b



# -------------------- App Layout --------------------



st.title("📷 Image Processing Toolkit — Streamlit + OpenCV")



# Menu bar simulation

menu = st.sidebar.selectbox("Menu", ["File", "Operations", "Help"])

if menu == 'File':

  st.sidebar.write("Use the upload control below to open an image, and Save after processing.")

elif menu == 'Help':

  st.sidebar.write("This app demonstrates many common image-processing operations using OpenCV. See resources in the README.")



# Upload

uploaded = st.sidebar.file_uploader("Open → Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"])



# Sidebar panels

st.sidebar.header("Operations")

show_info = st.sidebar.checkbox("Image Info", value=True)

conversion = st.sidebar.selectbox("Color Conversion", ["None", "BGR→RGB (swap)", "BGR→HSV (cv2)", "BGR→YCbCr (manual)", "BGR→Grayscale (cv2)", "Manual Gray"])

transform = st.sidebar.selectbox("Transformations", ["None", "Rotate", "Scale", "Translate", "Affine", "Perspective"]) 

filtering = st.sidebar.selectbox("Filtering & Morphology", ["None", "Smoothing", "Edge Filters", "Morphology"]) 

enhance = st.sidebar.selectbox("Enhancement / Edge Detection", ["None", "Histogram Eq", "CLAHE", "Sharpen", "Canny"])

compression = st.sidebar.selectbox("Compression / Save Format", ["None", "JPG", "PNG", "BMP"]) 



st.sidebar.markdown("---")

st.sidebar.write("Bonus & Tools")

split_mode = st.sidebar.checkbox("Comparison: Split (half original / half processed)", value=False)

show_slider_kernel = st.sidebar.slider("Kernel size (filters)", 1, 31, 3, step=2)

rotation_angle = st.sidebar.slider("Rotation angle", -180, 180, 0)

scale_factor = st.sidebar.slider("Scale factor (percent)", 10, 300, 100)

translate_x = st.sidebar.slider("Translate X (px)", -500, 500, 0)

translate_y = st.sidebar.slider("Translate Y (px)", -500, 500, 0)



# For affine/perspective, we'll provide example points

st.sidebar.markdown("**Affine / Perspective sample**")

use_sample_warp = st.sidebar.checkbox("Use sample warp points (auto)", value=True)



# Bitwise operations

bitwise = st.sidebar.selectbox("Bitwise Ops (requires 2nd image)", ["None", "AND", "OR", "XOR", "NOT"]) 

second_image = st.sidebar.file_uploader("(Optional) Second image for bitwise ops", type=["png", "jpg", "jpeg", "bmp"] )



# Compression quality if JPG

jpg_quality = st.sidebar.slider("JPG quality", 10, 100, 95)



# Camera input (bonus)

use_camera = st.sidebar.checkbox("Use Camera Input (capture image)")



# Layout columns for display area

col1, col2 = st.columns([1,1])



orig_placeholder = col1.empty()

proc_placeholder = col2.empty()



status = st.empty()



# Load image bytes

if uploaded is not None:

  file_bytes = uploaded.read()

  img_pil = Image.open(io.BytesIO(file_bytes))

  img_cv = pil_to_cv2(img_pil)

elif use_camera:

  cam_file = st.camera_input("Capture photo")

  if cam_file is not None:

    cf_bytes = cam_file.getvalue()

    img_pil = Image.open(io.BytesIO(cf_bytes))

    img_cv = pil_to_cv2(img_pil)

  else:

    img_cv = None

else:

  img_cv = None



if img_cv is None:

  st.info("Upload or capture an image to begin. (Use the sidebar controls)")

else:

  working = img_cv.copy()

  original_display = cv2_to_pil(img_cv)



  # Apply color conversion

  if conversion != 'None':

    if conversion == 'BGR→RGB (swap)':

      working = bgr_to_rgb_manual(working)

      # after manual swap, working is RGB — convert to BGR for subsequent ops

      working = cv2.cvtColor(working, cv2.COLOR_RGB2BGR)

    elif conversion == 'BGR→HSV (cv2)':

      hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)

      # show HSV as converted BGR for visualization (convert back to BGR via cvtColor)

      working = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif conversion == 'BGR→YCbCr (manual)':

      ycbcr = bgr_to_ycbcr_manual(working)

      # For visualization convert YCbCr to BGR using OpenCV conversion of placeholder

      working = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)

    elif conversion == 'BGR→Grayscale (cv2)':

      g = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

      working = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    elif conversion == 'Manual Gray':

      g = bgr_to_gray_manual(working)

      working = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)



  # Transformations

  if transform != 'None':

    if transform == 'Rotate':

      working = rotate_image(working, rotation_angle, scale=1.0)

    elif transform == 'Scale':

      sf = scale_factor / 100.0

      working = scale_image(working, sf, sf)

    elif transform == 'Translate':

      working = translate_image(working, translate_x, translate_y)

    elif transform == 'Affine':

      (h, w) = working.shape[:2]

      if use_sample_warp:

        pts1 = np.float32([[0,0], [w-1,0], [0,h-1]])

        pts2 = np.float32([[0, h*0.33], [w*0.85, h*0.25], [w*0.15, h*0.7]])

      else:

        pts1 = np.float32([[0,0], [w-1,0], [0,h-1]])

        pts2 = pts1

      working = affine_transform(working, pts1, pts2)

    elif transform == 'Perspective':

      (h, w) = working.shape[:2]

      if use_sample_warp:

        src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])

        dst = np.float32([[w*0.0,h*0.33],[w*0.85,h*0.1],[w*0.9,h*0.9],[w*0.1,h*0.85]])

      else:

        src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])

        dst = src

      working = perspective_transform(working, src, dst)



  # Filtering & Morphology

  if filtering != 'None':

    if filtering == 'Smoothing':

      working = apply_smoothing(working, method='gaussian', ksize=show_slider_kernel)

    elif filtering == 'Edge Filters':

      # choose Sobel by default

      working = apply_edge_filter(working, method='sobel')

    elif filtering == 'Morphology':

      working = apply_morphology(working, op='open', kernel_size=show_slider_kernel)



  # Enhancement & Edge Detection

  if enhance != 'None':

    if enhance == 'Histogram Eq':

      working = histogram_equalization(working)

    elif enhance == 'CLAHE':

      lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)

      l, a, b = cv2.split(lab)

      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

      cl = clahe.apply(l)

      limg = cv2.merge((cl,a,b))

      working = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    elif enhance == 'Sharpen':

      working = sharpen_image(working)

    elif enhance == 'Canny':

      working = apply_edge_filter(working, method='canny')



  # Bitwise ops

  if bitwise != 'None' and second_image is not None:

    sec_bytes = second_image.read()

    sec_pil = Image.open(io.BytesIO(sec_bytes))

    sec_cv = pil_to_cv2(sec_pil)

    working = bitwise_ops(working, sec_cv, op=bitwise.lower())

  elif bitwise == 'NOT':

    working = bitwise_ops(working, working, op='not')



  # Compression preview

  if compression != 'None':

    fmt = compression

    if fmt == 'None':

      pass

    else:

      save_fmt = 'JPG' if fmt == 'JPG' else fmt

      size, b = get_saved_size_and_bytes(working, fmt=save_fmt, quality=jpg_quality)

      if size is not None:

        status.write(f"Preview saved size as {save_fmt}: {size} bytes (quality={jpg_quality})")



  # Display

  proc_display = cv2_to_pil(working)



  if split_mode:

    # make split comparison image

    left = original_display.convert('RGB')

    right = proc_display.convert('RGB')

    w = left.width

    h = left.height

    # resize processed to original size

    right = right.resize((w, h))

    left_np = np.array(left)

    right_np = np.array(right)

    split = left_np.copy()

    split[:, :w//2, :] = right_np[:, :w//2, :]

    split_pil = Image.fromarray(split)

    orig_placeholder.image(split_pil, caption='Half Original / Half Processed (comparison)', use_column_width=True)

    proc_placeholder.empty()

  else:

    orig_placeholder.image(original_display, caption='Original Image', use_column_width=True)

    proc_placeholder.image(proc_display, caption='Processed Image', use_column_width=True)



  # Status bar details

  info = get_image_info(to_bytes(img_cv, fmt='PNG'))

  if info is not None:

    st.markdown("---")

    cols = st.columns(4)

    cols[0].metric("Dimensions (W x H)", f"{info['width']} x {info['height']}")

    cols[1].metric("Channels", f"{info['channels']}")

    cols[2].metric("Format", f"{info['format']}")

    cols[3].metric("Filesize", f"{info['size_bytes']} bytes")



  # Save functionality

  st.markdown("---")

  save_name = st.text_input("Save processed image as (filename without extension)", value=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

  save_fmt = st.selectbox("Save format", ["PNG","JPG","BMP"], index=0)

  if st.button("Save Processed Image"):

    pil_out = cv2_to_pil(working)

    out_buf = io.BytesIO()

    if save_fmt == 'JPG':

      pil_out.save(out_buf, format='JPEG', quality=jpg_quality)

    else:

      pil_out.save(out_buf, format=save_fmt)

    out_buf.seek(0)

    b = out_buf.read()

    # write to local file (optional)

    filename = f"{save_name}.{save_fmt.lower()}"

    with open(filename, 'wb') as f:

      f.write(b)

    st.success(f"Saved {filename} to working directory.")

    st.download_button("Download processed image", data=b, file_name=filename, mime=f"image/{save_fmt.lower()}")



  st.markdown("---")

  st.caption("Tips: Use the sidebar to chain operations. For precise transforms, prefer Affine or Perspective with custom points in code.")





# Footer

st.markdown("---")

st.write("Designed for Module 1 — Image Processing Fundamentals & Computer Vision")

