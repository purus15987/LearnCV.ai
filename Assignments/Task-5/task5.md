# 🧠 **Assignment 5 – Deep Learning Computer Vision Project with Real-time Deployment**  
---
## **Module**: Module 5 – Deep Learning in Computer Vision  
**Deadline**: Nov 14, 2025  
**Level**: 🟠 Intermediate → 🔴 Advanced  
**Points**: 100 points  
---
## **🎯 Objective**  
Design and implement a **full-stack Deep Learning-based Computer Vision system** that satisfies **real-time application** and **heavy-data analysis** requirements. The project must use **TensorFlow/Keras or PyTorch**, process **large-scale image datasets**, and be **deployable in real-time** via a web or local GUI (Streamlit/FastAPI/Flask).  

Students will **select, train, evaluate, optimize, and deploy** a DL model for a chosen CV task with **novelty**, **benchmarking**, and **measurable performance**.

---

## **📍 Project Selection & Core Requirements**  

| Requirement | Must Include |
|-----------|--------------|
| **Dataset** | Image-based, ≥1,000 labeled samples (e.g., COCO, Open Images, Kaggle, custom) |
| **Framework** | TensorFlow/Keras **OR** PyTorch |
| **Model** | Pre-trained (ResNet, YOLO, MobileNet, etc.) **OR** Custom CNN/Transformer |
| **Categories** | ✅ Real-time Application <br> ✅ Heavy-Data Analysis <br> + **One of**: Research-based **OR** Product/Prototype |
| **Deployment** | Real-time inference (webcam/RTSP/video file) |
| **GUI** | Streamlit / FastAPI / Flask |

---

## **🖼️ System Design (Streamlit + Real-time Pipeline)**  

### **1️⃣ Sidebar – Control Panel**  
```text
📂 Dataset & Model
   • Upload dataset (ZIP/CSV + images)
   • Select pre-trained model
   • Train / Load checkpoint

🎛️ Task Selection
   • Image Classification
   • Object Detection
   • Instance Segmentation
   • Pose Estimation

⚙️ Real-time Mode
   • Webcam / Video File / RTSP
   • FPS Display
   • Confidence Threshold Slider

📊 Evaluation
   • Show Metrics Table
   • Confusion Matrix / mAP Plot
   • Export Report
```

---

### **2️⃣ Main Area – Dual View**  
| Left Column | Right Column |
|------------|-------------|
| **Input Feed** (Live video / uploaded image) | **Output Overlay** (Bounding boxes, masks, labels, FPS) |

Below:  
📈 **Live Metrics Dashboard**  
- Accuracy / mAP@0.5  
- Inference Time (ms)  
- GPU/CPU Usage  
- Model Size (MB)  

---

## **🛠 Step-by-Step Implementation**

### **Phase 1 – Project Proposal & Setup**  
1. **Fill Google Form**: [Project Selection Form](https://docs.google.com/forms/d/e/1FAIpQLSdsMf7DfW_QhKf597f2dMstk03EIR-tLdQKbbdH07yFSTnp1A/viewform)  
2. **Submit Proposal**: Use [STARS Template](https://stars.iisc.ac.in/uploads/userfiles/files/STARS-proposal-template.docx)  
3. Initialize repo:  
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# OR
pip install tensorflow opencv-python streamlit albumentations wandb
```

---

### **Phase 2 – Dataset & Preprocessing**  
- Use **Albumentations** or **Torchvision transforms**  
- Implement **data augmentation** (flip, rotate, mosaic, mixup)  
- Create `data_loader.py` with train/val split  
- Log dataset stats (classes, samples, imbalance)  

---

### **Phase 3 – Model Development**  
Choose **one** task:  

#### **A. Image Classification**  
```python
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(2048, num_classes)
```

#### **B. Object Detection (YOLOv8 / Faster R-CNN)**  
```bash
pip install ultralytics
```
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # or yolov8s, yolov8m
```

#### **C. Segmentation (U-Net / SegFormer)**  
```python
from transformers import SegformerForSemanticSegmentation
```

---

### **Phase 4 – Training & Logging**  
- Use **Weights & Biases (wandb)** or **TensorBoard**  
- Log: loss curves, mAP, F1, learning rate  
- Save best checkpoint (`best.pt` or `.h5`)  

---

### **Phase 5 – Real-time Inference Engine**  
```python
cap = cv2.VideoCapture(0)  # or RTSP URL
while True:
    ret, frame = cap.read()
    results = model(frame, stream=True)
    annotated = results[0].plot()
    cv2.imshow("YOLO Real-time", annotated)
    if cv2.waitKey(1) == ord('q'): break
```

---

### **Phase 6 – Streamlit GUI Deployment**  
```python
# app.py
import streamlit as st
from inference import predict_image, predict_video

st.title("🚀 DL-CV Real-time System")
mode = st.sidebar.selectbox("Mode", ["Image", "Video", "Webcam"])

if mode == "Webcam":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        result = predict_image(frame, model)
        stframe.image(result, channels="BGR")
```

---

### **Phase 7 – Optimization & Edge Deployment**  
- Convert to **ONNX** or **TensorRT**  
```bash
python -m tf2onnx.convert --saved-model model/ --output model.onnx
```
- Measure **FPS on CPU/GPU/Jetson**  
- Export **model card** (size, params, latency)  

---

### **Phase 8 – Novelty & Benchmarking**  
| Task | Benchmark | Your Expected |
|------|----------|---------------|
| Classification | ResNet50: 76.1% (ImageNet) | ≥80% on custom data |
| Detection | YOLOv8n: 37.3 mAP (COCO) | ≥40 mAP + 30 FPS |
| Segmentation | DeepLabV3: 79.2% mIoU | ≥75% mIoU real-time |

**Add Novelty**:  
- Custom dataset (e.g., Indian traffic signs)  
- Hybrid model (CNN + ViT)  
- Low-light augmentation pipeline  
- Quantization-aware training  

---

## **📂 Deliverables**  

1. **Codebase**  
   - `app.py` (Streamlit GUI)  
   - `train.py`, `inference.py`, `utils.py`  
   - `requirements.txt`  
   - `model/` folder with `.pt` or `.h5`  

2. **Notebook**  
   - `DL_CV_Project_<roll_no>.ipynb`  
   - Training logs, visualizations, ablation study  

3. **Report (PDF)** – **STARS Proposal Format**  
   - Title, Abstract, Introduction  
   - Dataset, Methodology, Architecture Diagram  
   - Results (tables, graphs, confusion matrix)  
   - Real-time Demo Screenshots  
   - Benchmark Comparison  
   - Novel Contribution  
   - Future Work  

4. **Demo Video (< 3 min)**  
   - Show: Training → GUI → Real-time webcam inference  

5. **Google Form Submission**  
   - Link: [Submit Here](https://docs.google.com/forms/d/e/1FAIpQLSdsMf7DfW_QhKf597f2dMstk03EIR-tLdQKbbdH07yFSTnp1A/viewform)  

---

## **📊 Evaluation Rubric**  

| Criteria | Points | Description |
|--------|--------|-----------|
| **Project Proposal & Form** | 10 | Clarity, novelty, feasibility |
| **Dataset & Preprocessing** | 15 | Size, labeling, augmentation |
| **Model Implementation** | 20 | Correct architecture, training loop |
| **Real-time Performance** | 20 | FPS ≥ 15, low latency, deployment |
| **GUI & Usability** | 15 | Streamlit layout, interactivity |
| **Evaluation Metrics** | 10 | Proper metrics, visualization |
| **Report & Documentation** | 10 | STARS format, depth, visuals |
| **Novelty & Benchmark** | 10 | Unique element + comparison |
| **Total** | **100** | |

---

## **✅ Submission Instructions**  
1. Fork `learncv.ai` → `/assignments` branch  
2. Create folder: `/assignments/Task-5/<roll_no>/`  
3. Upload:  
   - `app.py`  
   - Notebook  
   - Report (`Proposal_<roll_no>.pdf`)  
   - `model/best.pt` or `.h5`  
   - `demo.mp4` (optional)  
4. Commit: `"Task 5: DL CV Project Submission"`  
5. Create **Pull Request** to `/assignments`  

---

## **📎 Resources**  
- [YOLOv8 Docs](https://docs.ultralytics.com/)  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)  
- [TensorFlow Model Zoo](https://github.com/tensorflow/models)  
- [PyTorch Vision](https://pytorch.org/vision/stable/index.html)  
- [Streamlit Deployment](https://docs.streamlit.io/)  
- [ONNX Runtime](https://onnxruntime.ai/)  

---

## **🚀 Bonus Add-Ons**  
- [x] **Mobile App** via TFLite + Flutter  
- [x] **Cloud Deployment** (Streamlit Cloud / AWS)  
- [x] **Multi-model Ensemble**  
- [x] **Active Learning Loop**  
- [x] **Explainability** (Grad-CAM, LIME)  

---

> ⚠️ **Note**: Plagiarism = **Zero**. Use pre-trained models **ethically** with proper citation.  
> Late submission: **-10% per day**

---