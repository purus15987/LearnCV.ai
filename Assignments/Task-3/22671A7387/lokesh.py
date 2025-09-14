import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os

# ------------------------ Helper Functions ------------------------

def load_image(image_file):
    image = Image.open(image_file)
    image = image.convert('RGB')
    return np.array(image)

def convert_color(image, option):
    if option == "RGB → BGR":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif option == "RGB → HSV":
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif option == "RGB → YCbCr":
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif option == "RGB → Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif option == "BGR → RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif option == "HSV → RGB":
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    elif option == "YCbCr → RGB":
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
    return image

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def scale_image(image, scale):
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w*scale), int(h*scale)))

def translate_image(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    h, w = image.shape[:2]
    return cv2.warpAffine(image, M, (w, h))

def affine_transform(image):
    h, w = image.shape[:2]
    pts1 = np.float32([[0,0], [w-1,0], [0,h-1]])
    pts2 = np.float32([[0,h*0.33], [w*0.85,h*0.25], [w*0.15,h*0.7]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, M, (w, h))

def perspective_transform(image):
    h, w = image.shape[:2]
    pts1 = np.float32([[0,0], [w-1,0], [0,h-1], [w-1,h-1]])
    pts2 = np.float32([[w*0.05,h*0.33], [w*0.9,h*0.1], [w*0.2,h*0.7], [w*0.8,h*0.9]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (w, h))

def apply_filter(image, filter_name, ksize=3):
    if filter_name == "Gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif filter_name == "Mean":
        return cv2.blur(image, (ksize, ksize))
    elif filter_name == "Median":
        return cv2.medianBlur(image, ksize)
    return image

def edge_detection(image, method, ksize=3, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == "Sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel = cv2.magnitude(sobelx, sobely)
        return np.uint8(sobel)
    elif method == "Laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return np.uint8(np.absolute(lap))
    elif method == "Canny":
        return cv2.Canny(gray, threshold1, threshold2)
    return gray

def morphological(image, operation, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    if operation == "Dilation":
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == "Erosion":
        return cv2.erode(image, kernel, iterations=1)
    elif operation == "Opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == "Closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

def histogram_equalization(image):
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        return cv2.equalizeHist(image)

def contrast_stretching(image):
    in_min = np.percentile(image, 5)
    in_max = np.percentile(image, 95)
    out_min = 0
    out_max = 255
    stretched = (image - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min
    stretched = np.clip(stretched, 0, 255)
    return stretched.astype(np.uint8)

def sharpening(image):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    return cv2.filter2D(image, -1, kernel)

def save_image(image, format, filename="output"):
    ext = format.lower()
    file_path = f"{filename}.{ext}"
    if ext == "jpg":
        cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return file_path

def get_image_info(image, image_file):
    h, w = image.shape[:2]
    c = 1 if len(image.shape) == 2 else image.shape[2]
    dpi = (72, 72)
    file_size = os.path.getsize(image_file) / 1024
    file_format = image_file.split('.')[-1].upper()
    return h, w, c, dpi, file_size, file_format

# ------------------------ Streamlit App ------------------------

st.title("🖼 Image Processing Toolkit")

# Sidebar
st.sidebar.title("Operations")
operation = st.sidebar.selectbox("Choose operation", [
    "None",
    "Image Info",
    "Color Conversions",
    "Transformations",
    "Filtering & Morphology",
    "Enhancement",
    "Edge Detection",
    "Compression"
])

# Upload Image
image_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"])
if image_file:
    original_image = load_image(image_file)
    processed_image = original_image.copy()

    # Image Info
    if operation == "Image Info":
        h, w, c, dpi, size, fmt = get_image_info(original_image, image_file.name)
        st.sidebar.write("*Image Details*")
        st.sidebar.write(f"Resolution: {w} × {h}")
        st.sidebar.write(f"Channels: {c}")
        st.sidebar.write(f"DPI: {dpi[0]}")
        st.sidebar.write(f"File Format: {fmt}")
        st.sidebar.write(f"File Size: {size:.2f} KB")

    # Color Conversions
    elif operation == "Color Conversions":
        color_option = st.sidebar.selectbox("Select Conversion", [
            "RGB → BGR", "RGB → HSV", "RGB → YCbCr", "RGB → Grayscale",
            "BGR → RGB", "HSV → RGB", "YCbCr → RGB"
        ])
        processed_image = convert_color(original_image, color_option)

    # Transformations
    elif operation == "Transformations":
        transform_type = st.sidebar.selectbox("Select Transformation", [
            "Rotation", "Scaling", "Translation", "Affine Transform", "Perspective Transform"
        ])
        if transform_type == "Rotation":
            angle = st.sidebar.slider("Angle", -180, 180, 0)
            processed_image = rotate_image(original_image, angle)
        elif transform_type == "Scaling":
            scale = st.sidebar.slider("Scale", 0.1, 3.0, 1.0, 0.1)
            processed_image = scale_image(original_image, scale)
        elif transform_type == "Translation":
            x_shift = st.sidebar.slider("Shift X", -100, 100, 0)
            y_shift = st.sidebar.slider("Shift Y", -100, 100, 0)
            processed_image = translate_image(original_image, x_shift, y_shift)
        elif transform_type == "Affine Transform":
            processed_image = affine_transform(original_image)
        elif transform_type == "Perspective Transform":
            processed_image = perspective_transform(original_image)

    # Filtering & Morphology
    elif operation == "Filtering & Morphology":
        filter_type = st.sidebar.selectbox("Select Filter", ["Gaussian", "Mean", "Median"])
        morph_type = st.sidebar.selectbox("Select Morphology", ["None", "Dilation", "Erosion", "Opening", "Closing"])
        ksize = st.sidebar.slider("Kernel Size", 3, 15, 3, 2)
        processed_image = apply_filter(original_image, filter_type, ksize)
        if morph_type != "None":
            processed_image = morphological(processed_image, morph_type, ksize)

    # Enhancement
    elif operation == "Enhancement":
        enh_type = st.sidebar.selectbox("Select Enhancement", ["Histogram Equalization", "Contrast Stretching", "Sharpening"])
        if enh_type == "Histogram Equalization":
            processed_image = histogram_equalization(original_image)
        elif enh_type == "Contrast Stretching":
            processed_image = contrast_stretching(original_image)
        elif enh_type == "Sharpening":
            processed_image = sharpening(original_image)

    # Edge Detection
    elif operation == "Edge Detection":
        edge_type = st.sidebar.selectbox("Select Edge Detection", ["Sobel", "Canny", "Laplacian"])
        ksize = st.sidebar.slider("Kernel Size", 3, 7, 3, 2)
        threshold1 = st.sidebar.slider("Threshold1", 50, 150, 100)
        threshold2 = st.sidebar.slider("Threshold2", 150, 300, 200)
        processed_image = edge_detection(original_image, edge_type, ksize, threshold1, threshold2)

    # Compression
    elif operation == "Compression":
        comp_format = st.sidebar.selectbox("Select Format", ["JPG", "PNG", "BMP"])
        file_path = save_image(original_image, comp_format, "compressed_image")
        size = os.path.getsize(file_path) / 1024
        st.sidebar.write(f"Saved as {comp_format} - Size: {size:.2f} KB")
        processed_image = original_image

    # Display Area
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original Image")
        st.image(original_image, use_column_width=True)
    with col2:
        st.header("Processed Image")
        st.image(processed_image, use_column_width=True)

    # Status Bar
    h, w = processed_image.shape[:2]
    c = 1 if len(processed_image.shape) == 2 else processed_image.shape[2]
    st.text(f"Dimensions: {w} × {h} × {c}")

    # Save Button
    if st.button("Save Processed Image"):
        save_format = st.selectbox("Save as", ["JPG", "PNG", "BMP"])
        save_file = save_image(processed_image, save_format, "saved_image")
        st.success(f"Image saved as {save_file}")

else:
    st.warning("Please upload an image to start.")