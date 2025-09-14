import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64

# ---------------- Helper Functions ----------------
def image_info(img, file_details=None):
    shape = img.shape
    H, W = shape[:2]
    C = shape[2] if len(shape) == 3 else 1
    info = f"{W} x {H}, Channels: {C}"
    if file_details:
        format = file_details['type'].split('/')[-1].upper()
        size = file_details['size'] // 1024
        info += f", Format: {format}, Size: {size} KB"
    return info

def get_image_array(uploaded_file):
    img = Image.open(uploaded_file)
    return np.array(img)

def convert_img(img, code):
    return cv2.cvtColor(img, code)

def apply_rotation(img, angle):
    H, W = img.shape[:2]
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (W, H))

def apply_scaling(img, factor):
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

def apply_translation(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    H, W = img.shape[:2]
    return cv2.warpAffine(img, M, (W, H))

def affine_transform(img, src, dst):
    M = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def perspective_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

def apply_filter(img, method, kernel=3):
    if method == "Gaussian":
        return cv2.GaussianBlur(img, (kernel, kernel), 0)
    elif method == "Median":
        return cv2.medianBlur(img, kernel)
    elif method == "Mean":
        return cv2.blur(img, (kernel, kernel))
    return img

def morphology(img, op, kernel=3):
    k = np.ones((kernel, kernel), np.uint8)
    if op == "Dilation":
        return cv2.dilate(img, k)
    elif op == "Erosion":
        return cv2.erode(img, k)
    elif op == "Opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
    elif op == "Closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    return img

def histogram_equalization(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    elif len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def contrast_stretching(img):
    a = np.min(img)
    b = np.max(img)
    stretched = (img - a) * (255 / (b - a))
    return np.uint8(stretched)

def sharpening(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def edge_detection(img, method, **kwargs):
    if method == "Sobel":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    elif method == "Canny":
        t1, t2 = kwargs.get("t1", 100), kwargs.get("t2", 200)
        return cv2.Canny(img, t1, t2)
    elif method == "Laplacian":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F)
    return img

def compress_image(img, method):
    ext = {"JPG": ".jpg", "PNG": ".png", "BMP": ".bmp"}[method]
    _, buf = cv2.imencode(ext, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf

def b64_file(buf, ext):
    b64 = base64.b64encode(buf).decode()
    href = f'<a href="data:file/{ext};base64,{b64}" download="processed.{ext}">Download processed image</a>'
    return href

# ---------------- Streamlit Layout ----------------
st.set_page_config(layout="wide")
st.title("🖼️ Image Processing & Analysis Toolkit")
st.caption("Assignment 3 • Python + OpenCV + Streamlit | All major CV toolkit features with GUI demo")

# Sidebar Menu
with st.sidebar:
    st.header("Operations Menu")
    st.markdown("**1. Image Info & Channel**")
    op = st.radio("Feature Category", [
        "Image Info", "Color Conversion", "Transformations",
        "Filtering & Morphology", "Enhancement", "Edge Detection",
        "Compression"
    ])
    uploaded = st.file_uploader("Open: Upload an image", type=['png', 'jpg', 'jpeg', 'bmp'])

# ---------------- Default / Uploaded Image ----------------
if uploaded is not None:
    # User uploaded an image
    file_details = {"filename": uploaded.name, "type": uploaded.type, "size": uploaded.size}
    img = get_image_array(uploaded)
else:
    # Default image (your KTM photo)
    st.info("No file uploaded. Using default image.")
    default_image_path = "my_photo.jpg"   # <- Make sure this file exists in project folder
    img = cv2.cvtColor(cv2.imread(default_image_path), cv2.COLOR_BGR2RGB)
    file_details = {"filename": "my_photo.jpg", "type": "jpg", "size": img.nbytes}

# ---------------- Processing ----------------
processed_img = img
st.markdown("---")

if op == "Image Info":
    st.write(image_info(img, file_details))
elif op == "Color Conversion":
    mode = st.selectbox("Color mode", [
        "RGB ↔ BGR", "RGB ↔ HSV", "RGB ↔ YCbCr", "RGB ↔ Grayscale"
    ])
    if mode == "RGB ↔ BGR":
        processed_img = convert_img(img, cv2.COLOR_RGB2BGR)
    elif mode == "RGB ↔ HSV":
        processed_img = convert_img(img, cv2.COLOR_RGB2HSV)
    elif mode == "RGB ↔ YCbCr":
        processed_img = convert_img(img, cv2.COLOR_RGB2YCrCb)
    elif mode == "RGB ↔ Grayscale":
        processed_img = convert_img(img, cv2.COLOR_RGB2GRAY)
elif op == "Transformations":
    ttype = st.selectbox("Type", ["Rotation", "Scaling", "Translation", "Affine", "Perspective"])
    if ttype == "Rotation":
        angle = st.slider("Rotation angle", -180, 180, 0)
        processed_img = apply_rotation(img, angle)
    elif ttype == "Scaling":
        factor = st.slider("Scale factor", 0.1, 3.0, 1.0)
        processed_img = apply_scaling(img, factor)
    elif ttype == "Translation":
        x = st.slider("Shift X", -100, 100, 0)
        y = st.slider("Shift Y", -100, 100, 0)
        processed_img = apply_translation(img, x, y)
    elif ttype == "Affine":
        st.info("Affine: uses 3 fixed points.")
        src = np.float32([[0, 0], [50, 50], [100, 10]])
        dst = np.float32([[10, 100], [20, 50], [100, 60]])
        processed_img = affine_transform(img, src, dst)
    elif ttype == "Perspective":
        st.info("Perspective: Simple fixed corners.")
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        dst = np.float32([[0, 0], [w - 1, 0], [int(w * 0.33), h - 1], [int(w * 0.66), h - 1]])
        processed_img = perspective_transform(img, src, dst)
elif op == "Filtering & Morphology":
    proctype = st.selectbox("Type", ["Gaussian", "Mean", "Median", "Dilation", "Erosion", "Opening", "Closing"])
    ksize = st.slider("Kernel size", 3, 15, 3, step=2)
    if proctype in ["Gaussian", "Mean", "Median"]:
        processed_img = apply_filter(img, proctype, ksize)
    else:
        processed_img = morphology(img, proctype, ksize)
elif op == "Enhancement":
    eff = st.selectbox("Type", ["Histogram Equalization", "Contrast Stretching", "Sharpening"])
    if eff == "Histogram Equalization":
        processed_img = histogram_equalization(img)
    elif eff == "Contrast Stretching":
        processed_img = contrast_stretching(img)
    elif eff == "Sharpening":
        processed_img = sharpening(img)
elif op == "Edge Detection":
    kind = st.selectbox("Type", ["Sobel", "Canny", "Laplacian"])
    if kind == "Canny":
        t1 = st.slider("Threshold 1", 0, 255, 100)
        t2 = st.slider("Threshold 2", 0, 255, 200)
        processed_img = edge_detection(img, "Canny", t1=t1, t2=t2)
    else:
        processed_img = edge_detection(img, kind)
elif op == "Compression":
    typ = st.selectbox("Save as", ["JPG", "PNG", "BMP"])
    buf = compress_image(img, typ)
    st.write(f"Compressed size: {len(buf) // 1024} KB")
    st.markdown(b64_file(buf, typ.lower()), unsafe_allow_html=True)

# ---------------- Display ----------------
st.markdown("### Display: Original vs Processed", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.image(img, caption="Original Image", channels="RGB")
with col2:
    if len(processed_img.shape) == 2:
        st.image(processed_img, caption="Processed Image", channels="GRAY")
    else:
        st.image(processed_img, caption="Processed Image", channels="RGB")

st.caption("Status: " + image_info(processed_img, file_details))

# ---------------- Save ----------------
if st.button("Save Processed Image"):
    buf = compress_image(processed_img, "PNG")
    b64 = base64.b64encode(buf).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="processed.png">Download processed image</a>'
    st.markdown(href, unsafe_allow_html=True)