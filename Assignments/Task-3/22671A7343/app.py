
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import io, os, time
from utils import (
    to_rgb, to_bgr, to_gray, to_hsv, to_ycrcb, rotate, scale, translate,
    affine_transform, perspective_transform, mean_filter, gaussian_filter,
    median_filter, sobel_edges, laplacian_edges, canny_edges, morphology,
    histogram_equalization, contrast_stretch, sharpen, bitwise_and, bitwise_or,
    bitwise_xor, bitwise_not, half_split_comparison
)

st.set_page_config(page_title="Image Processing & Analysis Toolkit", layout="wide")

st.title("📸 Image Processing & Analysis Toolkit")
st.caption("Module 1 – Image Processing Fundamentals & Computer Vision | Streamlit + OpenCV")

# ---- Simulated Menu Bar (File) ----
col_m1, col_m2, col_m3 = st.columns([1,1,6])
with col_m1:
    file_action = st.selectbox("File", ["Open", "Save Processed", "Exit"], label_visibility="collapsed")
with col_m2:
    st.write("")

# ---- Sidebar ----
st.sidebar.header("⚙️ Operations")
uploaded = st.sidebar.file_uploader("Open → Upload an Image", type=["png","jpg","jpeg","bmp","tiff"])
second_image = st.sidebar.file_uploader("Optional: Second Image (for bitwise ops)", type=["png","jpg","jpeg","bmp","tiff"])
show_status = st.sidebar.checkbox("Show Status Bar", value=True)
comparison_mode = st.sidebar.checkbox("🪟 Split Comparison (half original, half processed)", value=False)
enable_webcam = st.sidebar.checkbox("🎥 Real-time (Webcam) [Experimental]", value=False)

# ---- Placeholders ----
col1, col2 = st.columns(2)
orig_box = col1.empty()
proc_box = col2.empty()

# ---- Helpers ----
def load_image(file):
    img = Image.open(file)
    return img

def pil_to_bgr(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def bgr_to_pil(img_bgr):
    if img_bgr.ndim == 2:
        return Image.fromarray(img_bgr)
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def image_info(img_pil, raw_file):
    w, h = img_pil.size
    mode = img_pil.mode
    fmt = getattr(img_pil, "format", "N/A")
    dpi = img_pil.info.get("dpi", ("N/A","N/A"))
    size_bytes = None
    if raw_file is not None:
        try:
            size_bytes = len(raw_file.getbuffer())
        except Exception:
            pass
    return {
        "width": w, "height": h, "mode": mode, "format": fmt,
        "dpi": dpi, "size_bytes": size_bytes
    }

# ---- Default images ----
original_pil = None
processed_bgr = None
original_bgr = None

if uploaded is not None:
    original_pil = load_image(uploaded).convert("RGB")
    original_bgr = pil_to_bgr(original_pil)
else:
    st.info("Upload an image from the sidebar to get started.")
    original_pil = Image.new("RGB", (640, 420), color=(220, 220, 220))
    cv2.putText(np_array:=np.array(original_pil), "Upload an image →", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2, cv2.LINE_AA)
    original_pil = Image.fromarray(np_array)
    original_bgr = pil_to_bgr(original_pil)

# ---- Sidebar: Operation Categories ----
op_category = st.sidebar.selectbox(
    "Select Category",
    [
        "Image Info",
        "Color Conversions",
        "Transformations",
        "Filtering & Morphology",
        "Enhancement",
        "Edge Detection",
        "Compression",
        "Bitwise Operations"
    ]
)

# ---- Operation controls ----
if op_category == "Image Info":
    st.sidebar.subheader("📐 Image Properties")
    # nothing interactive

elif op_category == "Color Conversions":
    color_op = st.sidebar.radio("Conversion", ["RGB↔BGR","RGB→HSV","RGB→YCrCb","RGB→Grayscale"])

elif op_category == "Transformations":
    tfm = st.sidebar.radio("Transform", ["Rotation","Scaling","Translation","Affine","Perspective"])
    if tfm == "Rotation":
        angle = st.sidebar.slider("Angle (deg)", -180, 180, 30, 1)
        scale_v = st.sidebar.slider("Scale", 10, 300, 100, 5)/100.0
    elif tfm == "Scaling":
        fx = st.sidebar.slider("Scale X (%)", 10, 300, 120, 5)/100.0
        fy = st.sidebar.slider("Scale Y (%)", 10, 300, 120, 5)/100.0
    elif tfm == "Translation":
        tx = st.sidebar.slider("Shift X (px)", -300, 300, 40, 5)
        ty = st.sidebar.slider("Shift Y (px)", -300, 300, 40, 5)
    elif tfm == "Affine":
        st.sidebar.info("Using fixed reference triangles (demo).")
    elif tfm == "Perspective":
        st.sidebar.info("Using fixed source/destination quads (demo).")

elif op_category == "Filtering & Morphology":
    filt = st.sidebar.radio("Operation", ["Gaussian","Mean","Median","Sobel (edges)","Laplacian (edges)","Dilation","Erosion","Opening","Closing"])
    ksize = st.sidebar.slider("Kernel size", 3, 21, 5, 2)
    iters = st.sidebar.slider("Iterations (morphology)", 1, 5, 1, 1)

elif op_category == "Enhancement":
    enh = st.sidebar.radio("Enhance", ["Histogram Equalization","Contrast Stretching","Sharpening"])

elif op_category == "Edge Detection":
    edge = st.sidebar.radio("Detector", ["Sobel","Canny","Laplacian"])
    if edge == "Canny":
        t1 = st.sidebar.slider("Threshold1", 0, 255, 100, 1)
        t2 = st.sidebar.slider("Threshold2", 0, 255, 200, 1)

elif op_category == "Compression":
    fmt = st.sidebar.selectbox("Save format", ["JPG","PNG","BMP"])
    quality = st.sidebar.slider("Quality (JPG only)", 1, 100, 90, 1)

elif op_category == "Bitwise Operations":
    bop = st.sidebar.radio("Bitwise", ["AND","OR","XOR","NOT"])

# ---- Apply operation ----
processed_bgr = original_bgr.copy()

if op_category == "Image Info":
    pass

elif op_category == "Color Conversions":
    if color_op == "RGB↔BGR":
        rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        processed_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif color_op == "RGB→HSV":
        processed_bgr = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    elif color_op == "RGB→YCrCb":
        processed_bgr = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb)
    elif color_op == "RGB→Grayscale":
        processed_bgr = to_gray(original_bgr)

elif op_category == "Transformations":
    if tfm == "Rotation":
        processed_bgr = rotate(original_bgr, angle, scale=scale_v)
    elif tfm == "Scaling":
        processed_bgr = scale(original_bgr, fx=fx, fy=fy)
    elif tfm == "Translation":
        processed_bgr = translate(original_bgr, tx=tx, ty=ty)
    elif tfm == "Affine":
        h, w = original_bgr.shape[:2]
        src = np.float32([[0,0],[w-1,0],[0,h-1]])
        dst = np.float32([[0,0],[int(0.85*(w-1)), int(0.15*(h-1))],[int(0.15*(w-1)), int(0.85*(h-1))]])
        processed_bgr = affine_transform(original_bgr, src, dst)
    elif tfm == "Perspective":
        h, w = original_bgr.shape[:2]
        src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        dst = np.float32([[int(0.1*w),0],[int(0.9*w),0],[int(0.8*w),h-1],[int(0.2*w),h-1]])
        processed_bgr = perspective_transform(original_bgr, src, dst)

elif op_category == "Filtering & Morphology":
    gray = to_gray(original_bgr)
    if filt == "Gaussian":
        processed_bgr = gaussian_filter(original_bgr, ksize=ksize)
    elif filt == "Mean":
        processed_bgr = mean_filter(original_bgr, ksize=ksize)
    elif filt == "Median":
        processed_bgr = median_filter(original_bgr, ksize=ksize)
    elif filt == "Sobel (edges)":
        processed_bgr = sobel_edges(gray)
    elif filt == "Laplacian (edges)":
        processed_bgr = laplacian_edges(gray)
    elif filt in ["Dilation","Erosion","Opening","Closing"]:
        op_map = {"Dilation":"dilate","Erosion":"erode","Opening":"open","Closing":"close"}
        processed_bgr = morphology(gray, op=op_map[filt], ksize=ksize, iterations=iters)

elif op_category == "Enhancement":
    if enh == "Histogram Equalization":
        processed_bgr = histogram_equalization(original_bgr)
    elif enh == "Contrast Stretching":
        processed_bgr = contrast_stretch(original_bgr)
    elif enh == "Sharpening":
        processed_bgr = sharpen(original_bgr)

elif op_category == "Edge Detection":
    gray = to_gray(original_bgr)
    if edge == "Sobel":
        processed_bgr = sobel_edges(gray)
    elif edge == "Laplacian":
        processed_bgr = laplacian_edges(gray)
    elif edge == "Canny":
        processed_bgr = canny_edges(gray, t1=t1, t2=t2)

elif op_category == "Compression":
    # show compression result via download button
    pass

elif op_category == "Bitwise Operations":
    if bop == "NOT":
        processed_bgr = bitwise_not(original_bgr)
    else:
        if second_image is not None:
            sec_pil = load_image(second_image).convert("RGB")
            sec_bgr = cv2.cvtColor(np.array(sec_pil), cv2.COLOR_RGB2BGR)
        else:
            sec_bgr = original_bgr  # fallback
        if bop == "AND":
            processed_bgr = bitwise_and(original_bgr, sec_bgr)
        elif bop == "OR":
            processed_bgr = bitwise_or(original_bgr, sec_bgr)
        elif bop == "XOR":
            processed_bgr = bitwise_xor(original_bgr, sec_bgr)

# ---- Comparison mode ----
display_processed = processed_bgr
if comparison_mode and uploaded is not None:
    try:
        display_processed = half_split_comparison(original_bgr, processed_bgr if processed_bgr is not None else original_bgr)
    except Exception:
        display_processed = processed_bgr

# ---- Display images ----
orig_box.image(original_pil, caption="Original", use_container_width=True)
if processed_bgr is not None:
    proc_box.image(bgr_to_pil(display_processed), caption="Processed", use_container_width=True)
else:
    proc_box.image(original_pil, caption="Processed", use_container_width=True)

# ---- Status Bar ----
if show_status:
    info = image_info(original_pil, uploaded)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Dimensions (HxWxC)", f"{info['height']} x {info['width']} x {3 if original_bgr.ndim==3 else 1}")
    col_b.metric("DPI/PPI", f"{info['dpi']}")
    col_c.metric("Format", f"{info['format']}")
    if info['size_bytes'] is not None:
        col_d.metric("File size", f"{info['size_bytes']/1024:.2f} KB")
    else:
        col_d.metric("File size", "N/A")

# ---- File Actions ----
if file_action == "Save Processed":
    if processed_bgr is None:
        st.warning("No processed image to save.")
    else:
        # Provide download buttons for JPG/PNG/BMP regardless of chosen op
        pil_out = bgr_to_pil(processed_bgr)
        buf_png = io.BytesIO()
        pil_out.save(buf_png, format="PNG")
        st.download_button("⬇️ Download PNG", buf_png.getvalue(), file_name="processed.png", mime="image/png")

        buf_jpg = io.BytesIO()
        pil_out.save(buf_jpg, format="JPEG", quality=95)
        st.download_button("⬇️ Download JPG", buf_jpg.getvalue(), file_name="processed.jpg", mime="image/jpeg")

        buf_bmp = io.BytesIO()
        pil_out.save(buf_bmp, format="BMP")
        st.download_button("⬇️ Download BMP", buf_bmp.getvalue(), file_name="processed.bmp", mime="image/bmp")

if op_category == "Compression" and uploaded is not None:
    # Allow user to export in selected format and show final size
    pil_out = bgr_to_pil(processed_bgr)
    if fmt == "JPG":
        buf = io.BytesIO()
        pil_out.save(buf, format="JPEG", quality=quality, optimize=True)
        size_kb = len(buf.getvalue())/1024
        st.download_button(f"⬇️ Save JPG (Q={quality}) — {size_kb:.2f} KB", buf.getvalue(), file_name=f"compressed_q{quality}.jpg", mime="image/jpeg")
    elif fmt == "PNG":
        buf = io.BytesIO()
        pil_out.save(buf, format="PNG", optimize=True)
        size_kb = len(buf.getvalue())/1024
        st.download_button(f"⬇️ Save PNG — {size_kb:.2f} KB", buf.getvalue(), file_name="compressed.png", mime="image/png")
    elif fmt == "BMP":
        buf = io.BytesIO()
        pil_out.save(buf, format="BMP")
        size_kb = len(buf.getvalue())/1024
        st.download_button(f"⬇️ Save BMP — {size_kb:.2f} KB", buf.getvalue(), file_name="compressed.bmp", mime="image/bmp")

# ---- Webcam (experimental) ----
if enable_webcam:
    st.info("Experimental: will attempt to open your default webcam and show processed frames for ~10 seconds.")
    if st.button("Start Webcam Demo"):
        placeholder = st.empty()
        import time
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam.")
        else:
            start = time.time()
            while time.time() - start < 10:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_bgr = frame
                show = frame_bgr
                # simple demo: apply selected op to live frame
                if op_category == "Edge Detection":
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    if edge == "Canny":
                        show = canny_edges(gray, t1=t1, t2=t2)
                    elif edge == "Sobel":
                        show = sobel_edges(gray)
                    else:
                        show = laplacian_edges(gray)
                elif op_category == "Enhancement":
                    if enh == "Sharpening":
                        show = sharpen(frame_bgr)
                    elif enh == "Histogram Equalization":
                        show = histogram_equalization(frame_bgr)
                    else:
                        show = contrast_stretch(frame_bgr)
                placeholder.image(show, channels="BGR", use_container_width=True)
            cap.release()
        st.success("Webcam demo ended.")

if file_action == "Exit":
    st.stop()
