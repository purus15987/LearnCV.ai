import streamlit as st
import cv2
import numpy as np
from skimage import exposure
from io import BytesIO
from PIL import Image

st.set_page_config(layout="wide", page_title="Image Processing Toolkit")


def image_info(img, fname):
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    size = img.nbytes / 1024
    fmt = fname.split(".")[-1].upper()
    return f"{w}×{h} px | {channels} channels | {fmt} | {size:.1f} KB"


def show_image(img, caption=""):
    buf = BytesIO()
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    im.save(buf, format="PNG")
    st.image(buf.getvalue(), caption=caption, use_column_width=True)


def convert_color(img, code):
    return cv2.cvtColor(img, code)


def apply_filter(img, kind):
    if kind == "Gaussian":
        return cv2.GaussianBlur(img, (7, 7), 0)
    elif kind == "Median":
        return cv2.medianBlur(img, 7)
    elif kind == "Mean":
        return cv2.blur(img, (7, 7))
    return img


def enhance_image(img, kind):
    if kind == "Histogram Equalization":
        if len(img.shape) == 2:
            return cv2.equalizeHist(img)
        else:
            # Convert to YCrCb and equalize Y channel
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    elif kind == "Contrast Stretching":
        p2, p98 = np.percentile(img, (2, 98))
        img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
        return (
            (img_rescale * 255).astype(np.uint8)
            if img_rescale.max() <= 1.0
            else img_rescale
        )
    elif kind == "Sharpening":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    return img


def edge_detect(img, method):
    if method == "Sobel":
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        abs_grad = cv2.convertScaleAbs(grad_x) + cv2.convertScaleAbs(grad_y)
        return abs_grad
    elif method == "Canny":
        return cv2.Canny(img, 100, 200)
    elif method == "Laplacian":
        lap = cv2.Laplacian(img, cv2.CV_64F)
        return cv2.convertScaleAbs(lap)
    return img


def apply_morphology(img, func):
    kernel = np.ones((5, 5), np.uint8)
    if func == "Dilation":
        return cv2.dilate(img, kernel, iterations=1)
    elif func == "Erosion":
        return cv2.erode(img, kernel, iterations=1)
    elif func == "Opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif func == "Closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def compress_image(img, fmt):
    buf = BytesIO()
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    im.save(buf, format=fmt)
    size = buf.tell() / 1024
    return buf.getvalue(), size


# --- Layout ---
st.title("🖼️ Image Processing & Analysis Toolkit")
st.markdown(
    "A visual playground for classical CV operations. Upload an image, select operations, and compare results side-by-side."
)

menu = st.sidebar.radio("Menu", ["Open/Upload Image", "Exit"])
if menu == "Exit":
    st.write("App Closed.")
    st.stop()

uploaded_file = st.sidebar.file_uploader(
    "Choose an image (jpg/png/bmp)", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    orig_img = img.copy()
    op_category = st.sidebar.selectbox(
        "Operation Category",
        [
            "Image Info",
            "Color Conversion",
            "Transformations",
            "Filtering/Morphology",
            "Enhancement",
            "Edge Detection",
            "Compression",
        ],
    )

    processed_img = img.copy()
    if op_category == "Image Info":
        st.sidebar.markdown("---")
        st.sidebar.text(image_info(img, uploaded_file.name))
        ch = img.shape[2] if len(img.shape) == 3 else 1
        st.sidebar.markdown(f"Channels: {ch}")
        st.sidebar.markdown(f"Shape: {img.shape}")
    elif op_category == "Color Conversion":
        conversions = {
            "RGB ↔ BGR": cv2.COLOR_BGR2RGB,
            "RGB ↔ HSV": cv2.COLOR_BGR2HSV,
            "RGB ↔ YCbCr": cv2.COLOR_BGR2YCrCb,
            "RGB ↔ Grayscale": cv2.COLOR_BGR2GRAY,
        }
        op = st.sidebar.selectbox("Conversion", list(conversions.keys()))
        processed_img = convert_color(img, conversions[op])
        if op == "RGB ↔ Grayscale":
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    elif op_category == "Transformations":
        method = st.sidebar.selectbox(
            "Method", ["Rotation", "Scaling", "Translation", "Affine", "Perspective"]
        )
        if method == "Rotation":
            angle = st.sidebar.slider("Angle", -180, 180, 45)
            M = cv2.getRotationMatrix2D(
                (img.shape[1] // 2, img.shape[0] // 2), angle, 1
            )
            processed_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            processed_img = cv2.warpAffine(img, M, (img.shape[1], img.shape))
        elif method == "Scaling":
            scale = st.sidebar.slider("Scale", 10, 300, 100)
            processed_img = cv2.resize(img, None, fx=scale / 100, fy=scale / 100)
        elif method == "Translation":
            dx = st.sidebar.slider("Shift X", -200, 200, 50)
            dy = st.sidebar.slider("Shift Y", -200, 200, 50)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            processed_img = cv2.warpAffine(img, M, (img.shape[1], img.shape))
        elif method == "Affine":
            st.sidebar.write("Uses default 3-point affine transform")
            rows, cols, ch = img.shape
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
            M = cv2.getAffineTransform(pts1, pts2)
            processed_img = cv2.warpAffine(img, M, (cols, rows))
        elif method == "Perspective":
            st.sidebar.write("Uses default 4-point perspective transform")
            rows, cols, ch = img.shape
            pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
            pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            processed_img = cv2.warpPerspective(img, M, (300, 300))
    elif op_category == "Filtering/Morphology":
        method = st.sidebar.selectbox(
            "Operation",
            [
                "Gaussian",
                "Median",
                "Mean",
                "Sobel",
                "Laplacian",
                "Dilation",
                "Erosion",
                "Opening",
                "Closing",
            ],
        )
        if method in ["Gaussian", "Median", "Mean"]:
            processed_img = apply_filter(img, method)
        elif method in ["Sobel", "Laplacian"]:
            processed_img = edge_detect(img, method)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
            processed_img = apply_morphology(binary, method)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    elif op_category == "Enhancement":
        enh = st.sidebar.selectbox(
            "Enhance", ["Histogram Equalization", "Contrast Stretching", "Sharpening"]
        )
        processed_img = enhance_image(img, enh)
    elif op_category == "Edge Detection":
        edge = st.sidebar.selectbox("Detector", ["Sobel", "Canny", "Laplacian"])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = edge_detect(gray, edge)
        processed_img = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif op_category == "Compression":
        fmt = st.sidebar.selectbox("Format", ["JPEG", "PNG", "BMP"])
        data, fsize = compress_image(img, fmt)
        st.sidebar.markdown(f"Compressed file size: {fsize:.1f} KB")
        st.sidebar.download_button(f"Download {fmt}", data, f"processed.{fmt.lower()}")
    # --- Layout
    c1, c2 = st.columns(2)
    with c1:
        show_image(orig_img, "Original Image")
    with c2:
        show_image(processed_img, "Processed Image")
    st.markdown("---")
    st.text("Status Bar")
    ch = img.shape[2] if len(img.shape) == 3 else 1
    st.text(
        f"Dimensions: {img.shape}x{img.shape[1]}, Channels: {ch}, Format: {uploaded_file.name.split('.')[-1].upper()}"
    )
    st.text(f"File size: {img.nbytes / 1024:.1f} KB")
    st.button("Save Processed Image", key="savebtn")
else:
    st.info("Upload an image from the sidebar to begin.")
