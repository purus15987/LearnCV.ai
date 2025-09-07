#type: ignore

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

# ---------------------- Helper Functions ----------------------

def load_image(image_file):
    image = Image.open(image_file)
    image = image.convert("RGB")
    return np.array(image)

def convert_color(image, mode):
    if mode == "RGB → BGR":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif mode == "RGB → HSV":
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif mode == "RGB → YCbCr":
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif mode == "RGB → Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif mode == "BGR → RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif mode == "HSV → RGB":
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    elif mode == "YCbCr → RGB":
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
    else:
        return image

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def scale_image(image, scale):
    (h, w) = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h))

def translate_image(image, tx, ty):
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    (h, w) = image.shape[:2]
    return cv2.warpAffine(image, matrix, (w, h))

def apply_filter(image, filter_type, ksize):
    if filter_type == "Gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif filter_type == "Mean":
        return cv2.blur(image, (ksize, ksize))
    elif filter_type == "Median":
        return cv2.medianBlur(image, ksize)

def edge_detection(image, method, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == "Sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(sobel)
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
    elif method == "Laplacian":
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
    elif method == "Canny":
        edges = cv2.Canny(gray, threshold1, threshold2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def histogram_equalization(image):
    if len(image.shape) == 2:
        equ = cv2.equalizeHist(image)
    else:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        equ = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return equ

def contrast_stretching(image):
    in_min = np.min(image)
    in_max = np.max(image)
    out_min = 0
    out_max = 255
    stretched = ((image - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min).astype(np.uint8)
    return stretched

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(image, -1, kernel)
    return sharp

def apply_morphology(image, operation, ksize):
    kernel = np.ones((ksize, ksize), np.uint8)
    if operation == "Dilation":
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == "Erosion":
        return cv2.erode(image, kernel, iterations=1)
    elif operation == "Opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == "Closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def save_image(img, filename, format):
    is_success, buffer = cv2.imencode(f'.{format}', img)
    if is_success:
        with open(filename, 'wb') as f:
            f.write(buffer)
    return filename

def get_image_info(image, image_file=None):
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    format = image_file.type if image_file else "N/A"
    size_kb = round(len(image.tobytes()) / 1024, 2)
    return {"Height": h, "Width": w, "Channels": channels, "Format": format, "Size": f"{size_kb} KB"}

# ---------------------- Streamlit App ----------------------

st.set_page_config(page_title="Image Processing Toolkit", layout="wide")
st.title("🖼 Image Processing Toolkit")

# Menu Bar
menu = st.sidebar.selectbox("Select Menu", ["File", "Operations"])

if menu == "File":
    st.sidebar.subheader("File Menu")
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg', 'bmp'])
    save_option = st.sidebar.button("Save Processed Image")
    exit_option = st.sidebar.button("Exit")

else:
    uploaded_file = None
    save_option = False
    exit_option = False

if uploaded_file is not None:
    image = load_image(uploaded_file)
    processed_image = image.copy()
    info = get_image_info(image, uploaded_file)

    # Sidebar Operations
    st.sidebar.subheader("Image Info")
    if st.sidebar.button("Show Info"):
        st.sidebar.write(f"Resolution: {info['Height']} x {info['Width']}")
        st.sidebar.write(f"Channels: {info['Channels']}")
        st.sidebar.write(f"File Format: {info['Format']}")
        st.sidebar.write(f"File Size: {info['Size']}")

    st.sidebar.subheader("Color Conversions")
    color_op = st.sidebar.selectbox("Choose Conversion", ["None", "RGB → BGR", "RGB → HSV", "RGB → YCbCr", "RGB → Grayscale"])
    if color_op != "None":
        processed_image = convert_color(processed_image, color_op)

    st.sidebar.subheader("Transformations")
    transform_op = st.sidebar.selectbox("Choose Transform", ["None", "Rotation", "Scaling", "Translation"])
    if transform_op == "Rotation":
        angle = st.sidebar.slider("Angle", -180, 180, 0)
        processed_image = rotate_image(processed_image, angle)
    elif transform_op == "Scaling":
        scale = st.sidebar.slider("Scale Factor", 0.1, 3.0, 1.0)
        processed_image = scale_image(processed_image, scale)
    elif transform_op == "Translation":
        tx = st.sidebar.slider("Shift X", -100, 100, 0)
        ty = st.sidebar.slider("Shift Y", -100, 100, 0)
        processed_image = translate_image(processed_image, tx, ty)

    st.sidebar.subheader("Filtering")
    filter_op = st.sidebar.selectbox("Choose Filter", ["None", "Gaussian", "Mean", "Median"])
    if filter_op != "None":
        ksize = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
        processed_image = apply_filter(processed_image, filter_op, ksize)

    st.sidebar.subheader("Edge Detection")
    edge_op = st.sidebar.selectbox("Choose Edge Filter", ["None", "Sobel", "Laplacian", "Canny"])
    if edge_op != "None":
        if edge_op == "Canny":
            t1 = st.sidebar.slider("Threshold 1", 0, 500, 100)
            t2 = st.sidebar.slider("Threshold 2", 0, 500, 200)
            processed_image = edge_detection(processed_image, edge_op, t1, t2)
        else:
            processed_image = edge_detection(processed_image, edge_op)

    st.sidebar.subheader("Enhancement")
    enhance_op = st.sidebar.selectbox("Choose Enhancement", ["None", "Histogram Equalization", "Contrast Stretching", "Sharpening"])
    if enhance_op == "Histogram Equalization":
        processed_image = histogram_equalization(processed_image)
    elif enhance_op == "Contrast Stretching":
        processed_image = contrast_stretching(processed_image)
    elif enhance_op == "Sharpening":
        processed_image = sharpen_image(processed_image)

    st.sidebar.subheader("Morphology")
    morph_op = st.sidebar.selectbox("Choose Morph Operation", ["None", "Dilation", "Erosion", "Opening", "Closing"])
    if morph_op != "None":
        ksize = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
        processed_image = apply_morphology(processed_image, morph_op, ksize)

    # Save functionality
    if save_option:
        filename = f"processed_image.png"
        save_image(processed_image, filename, "png")
        st.sidebar.success(f"Image saved as {filename}")

    # Exit functionality
    if exit_option:
        st.sidebar.warning("Exiting...")
        st.stop()

    # Display Area
    st.subheader("Original vs Processed Image")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        st.image(processed_image, caption="Processed Image", use_column_width=True)

    # Status Bar
    h, w = processed_image.shape[:2]
    channels = processed_image.shape[2] if len(processed_image.shape) == 3 else 1
    st.markdown(f"**Image Properties:** Dimensions: {w}x{h} | Channels: {channels} | Format: {info['Format']}")

else:
    st.info("Upload an image to start processing.")
