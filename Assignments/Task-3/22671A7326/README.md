- Name: K.Varshitha
- RollNo: 22671A7326
- Branch: AIML-A
#

# 🖼️ Image Processing & Analysis Toolkit

This project is a **Streamlit-based web application** designed as an educational and experimental platform for learning and applying classical computer vision (CV) techniques. With this app, users can upload images, apply a variety of transformations, filters, enhancements, and compression methods, and instantly view results side by side with the original image.  

It bridges the gap between theoretical concepts in image processing and practical, hands-on experimentation — making it useful for students, developers, and researchers who want to explore core CV techniques without writing lengthy code themselves.

---

## 🎯 Project Objectives

- To provide an **interactive playground** for testing computer vision operations.  
- To demonstrate the functionality of **OpenCV** and related libraries in a user-friendly interface.  
- To help learners understand how different **filters, transformations, and enhancements** affect digital images.  
- To create a **ready-to-deploy app** that can run locally or on platforms like Hugging Face Spaces.  

---

## 🛠️ Tech Stack

- **Streamlit** → For building the interactive web interface.  
- **OpenCV (cv2)** → Core image processing operations (color conversion, filtering, edge detection, morphology).  
- **NumPy** → Efficient numerical operations and array manipulations.  
- **scikit-image (exposure module)** → Used for intensity rescaling and contrast stretching.  
- **Pillow (PIL)** → For image handling and saving in various formats.  

System dependencies:
- `libgl1` and `libglib2.0-0` are included in `packages.txt` to support OpenCV’s GUI functions on Linux servers such as Hugging Face Spaces.

---

## 🚀 Features in Detail

### 1. Image Information
- Extracts and displays metadata such as:
  - Resolution (width × height)
  - Number of channels (grayscale or RGB)
  - File format (JPEG, PNG, BMP, etc.)
  - File size in KB  
- Useful for validating datasets and understanding image properties before processing.

---

### 2. Color Conversion
- Supports common color space conversions:
  - **RGB ↔ BGR** (swapping OpenCV’s default with standard RGB order)
  - **RGB ↔ HSV** (Hue, Saturation, Value model for color analysis)
  - **RGB ↔ YCbCr** (luminance and chrominance separation, common in video)
  - **RGB ↔ Grayscale** (reduces channels to intensity only)  
- Converts images and reverts grayscale images into 3-channel format for consistency.

---

### 3. Geometric Transformations
- **Rotation** → Rotate images at user-defined angles.  
- **Scaling** → Zoom in/out with a scale slider.  
- **Translation** → Shift the image along X and Y axes.  
- **Affine Transform** → Applies linear mapping using three-point correspondences.  
- **Perspective Transform** → Simulates viewpoint changes using four-point mapping.  

These operations demonstrate how images can be manipulated in terms of orientation and geometry.

---

### 4. Filtering & Morphological Operations
- **Smoothing Filters**:
  - Gaussian blur
  - Median filter
  - Mean filter  
  These are useful for noise reduction and pre-processing.  

- **Edge-based Filters**:
  - Sobel operator
  - Laplacian operator  

- **Morphological Operations**:
  - Dilation
  - Erosion
  - Opening
  - Closing  
  These are applied to binary images to manipulate structures, useful in object detection and segmentation.

---

### 5. Image Enhancement
- **Histogram Equalization** → Improves contrast by redistributing intensity values.  
- **Contrast Stretching** → Enhances global contrast by rescaling pixel intensity ranges.  
- **Sharpening** → Uses convolution kernels to enhance edges and details.

---

### 6. Edge Detection
- Provides three popular operators:
  - **Sobel** → Gradient-based detection in horizontal and vertical directions.  
  - **Canny** → Multi-stage algorithm, widely used for robust edge detection.  
  - **Laplacian** → Captures regions of rapid intensity change.  

---

### 7. Compression & Export
- Users can compress images into:
  - JPEG
  - PNG
  - BMP  
- Displays the compressed file size and allows direct download of the processed file.

---

### 8. Comparison View
- Side-by-side layout with **Original Image** on the left and **Processed Image** on the right.  
- Provides a **Status Bar** showing:
  - Dimensions
  - Number of channels
  - Format
  - File size  
- Makes it easier to compare effects in real time.
