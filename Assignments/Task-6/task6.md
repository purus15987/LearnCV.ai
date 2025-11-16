# **Assignment 6 – Classical Feature Engineering for Your Selected DL-CV Project**  
---
## **Module**: Module 5 – Bridging Classical & Deep Learning in Computer Vision  
**Deadline**: Dec 1, 2025  
**Level**: 🟠 Intermediate → 🔴 Advanced  
**Points**: 100 points  
---
## **🎯 Objective**  
**Enhance your Assignment 5 Deep Learning project** by integrating **classical (non-deep) feature engineering techniques** to:  
- Improve **robustness**  
- Reduce **data dependency**  
- Enable **hybrid models** (Classical + DL)  
- Support **low-data regimes** or **edge deployment**  

You will **analyze, implement, and compare** classical handcrafted features **alongside your DL pipeline** from Assignment 5.

---

## **📍 Project Context (Link to Assignment 5)**  
| Your Assignment 5 Project | Must Be Used Here |
|--------------------------|-------------------|
| Dataset (≥1,000 images) | Same |
| Task (Classification / Detection / Segmentation) | Same |
| Real-time GUI (Streamlit) | Extended |
| DL Model (YOLO, ResNet, etc.) | Baseline |

---

## **🖼️ System Design – Hybrid Pipeline (Classical + DL)**  

```text
┌─────────────────┐
│   Input Image   │
└───────┬─────────┘
        │
        ▼
┌─────────────────┐     ┌─────────────────────┐
│ Classical Path  │────▶│  Feature Fusion     │
│ (HOG, SIFT, etc)│     │  (Concat / MLP)     │
└───────┬─────────┘     └──────────┬──────────┘
        │                        │
        ▼                        ▼
┌─────────────────┐     ┌─────────────────────┐
│   DL Path       │     │   Hybrid Model      │
│ (CNN Backbone)  │     │   (Early/Late Fusion)│
└───────┬─────────┘     └──────────┬──────────┘
        │                        │
        ▼                        ▼
┌─────────────────┐     ┌─────────────────────┐
│  Classical Only │     │   Final Prediction  │
│   Classifier    │     │   (Real-time Output)│
└─────────────────┘     └─────────────────────┘
```

---

## **🛠 Step-by-Step Implementation**

---

### **Phase 1 – Classical Feature Extraction Module**

Implement **at least 3** classical techniques based on your **Assignment 5 task**:

| Task | Recommended Classical Features |
|------|--------------------------------|
| **Classification** | HOG, LBP, Color Histograms, SIFT + BoW |
| **Object Detection** | HOG + Sliding Window, Edge Maps, Haar Cascades |
| **Segmentation** | Texture (GLCM), Superpixels (SLIC), Watershed |

#### **Example: HOG + SVM (Classical Baseline)**
```python
from skimage.feature import hog
from sklearn.svm import SVC

def extract_hog(img):
    return hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2))

# Extract from dataset
X_hog = [extract_hog(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in images]
clf = SVC(kernel='rbf').fit(X_hog, labels)
```

---

### **Phase 2 – Hybrid Feature Fusion**

#### **Option A: Late Fusion (Prediction Level)**
```python
dl_pred = model_dl(image)        # [0.9, 0.1]
classical_pred = clf.predict(hog_features)  # [1]

final_pred = weighted_avg(dl_pred, classical_pred, w=0.7)
```

#### **Option B: Early Fusion (Feature Level)**
```python
dl_features = cnn_backbone(image)      # (1, 2048)
hog_features = extract_hog(image)      # (3780,)

fused = np.concatenate([dl_features, hog_features])
out = fusion_mlp(fused)
```

---

### **Phase 3 – Real-time GUI Extension (Streamlit)**

#### **New Sidebar Panel: "Classical Mode"**
```python
st.sidebar.header("🛠 Classical Pipeline")
use_classical = st.checkbox("Enable Classical Features")
method = st.selectbox("Feature", ["HOG", "LBP", "SIFT+BoW", "GLCM"])

if use_classical:
    st.write(f"Using {method} + SVM")
    show_feature_map = st.checkbox("Show Feature Visualization")
```

#### **Main Panel – Triple View**
| Left | Center | Right |
|------|--------|-------|
| **Input** | **Classical Features** (HOG viz, LBP map) | **DL Output + Hybrid** |

---

### **Phase 4 – Benchmarking: Classical vs DL vs Hybrid**

| Metric | Classical | DL (Assg 5) | Hybrid |
|--------|---------|-------------|--------|
| Accuracy / mAP | ? | ≥80% | **+2–5%** |
| FPS (CPU) | 40 | 15 | 25 |
| Model Size | <1 MB | 100+ MB | <110 MB |
| Training Data Needed | 100 | 1000 | 500 |

---

### **Phase 5 – Novelty via Classical Engineering**

Add **one unique classical contribution**:

| Idea | Description |
|------|-----------|
| **Adaptive HOG** | Dynamic cell size based on object scale |
| **Multi-resolution LBP** | Pyramid of LBP for scale invariance |
| **Color + Texture Fusion** | HSV Hist + GLCM entropy |
| **Edge-Guided Superpixels** | Use Canny + SLIC for better segmentation |
| **Haar + YOLO Ensemble** | Face detection fallback when YOLO fails |

---

## **📂 Deliverables**

1. **Codebase**  
   - `classical/` folder  
     - `hog_extractor.py`  
     - `lbp.py`, `sift_bow.py`  
     - `svm_classifier.py`  
   - Updated `app.py` with **Classical Mode toggle**  
   - `hybrid_fusion.py`  

2. **Notebook**  
   - `Classical_vs_DL_<roll_no>.ipynb`  
   - Compare feature visualizations  
   - Ablation: Classical only → DL only → Hybrid  

3. **Report (PDF)** – **5–7 pages**  
   - Introduction: Why classical still matters?  
   - Feature Engineering Pipeline (diagrams)  
   - Mathematical formulations (HOG, LBP, GLCM)  
   - Comparative Results (tables, graphs)  
   - Real-time FPS & memory profiling  
   - Failure cases: When classical helps DL  
   - Novel Contribution  

4. **Demo Video (< 2 min)**  
   - Show:  
     1. DL-only inference  
     2. Classical-only (HOG + SVM)  
     3. Hybrid (faster + accurate)  
     4. Feature visualization toggle  

---

## **📊 Evaluation Rubric**

| Criteria | Points | Description |
|--------|--------|-----------|
| **Classical Implementation** | 25 | ≥3 methods, correct, modular |
| **Hybrid Fusion Logic** | 20 | Early or late fusion, justified |
| **GUI Integration** | 15 | Toggle, visualization, real-time |
| **Performance Comparison** | 20 | Tables, graphs, FPS, accuracy |
| **Novelty & Analysis** | 10 | Unique classical trick + insight |
| **Report & Notebook** | 10 | Clarity, math, visuals |
| **Total** | **100** | |

---

## **✅ Submission Instructions**  
1. Fork `learncv.ai` → `/assignments` branch  
2. Create folder: `/assignments/Task-6/<roll_no>/`  
3. Upload:  
   - Updated `app.py`  
   - `classical/` module  
   - Notebook  
   - Report (`Classical_Hybrid_<roll_no>.pdf`)  
   - `demo_classical.mp4`  
4. Commit: `"Task 6: Classical Feature Engineering"`  
5. **Pull Request** to `/assignments`  

---

## **📎 Resources**  
- [Scikit-Image Feature Docs](https://scikit-image.org/docs/stable/api/api.html#feature)  
- [HOG Paper – Dalal & Triggs (2005)](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)  
- [LBP Tutorial](https://www.pyimagesearch.com/2021/06/07/local-binary-patterns-with-python-opencv/)  
- [BoW + SIFT](https://opencv-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html)  
- [GLCM Texture](https://www.youtube.com/watch?v=Upbduu4nEiA)  

---

## **🚀 Bonus Add-Ons**  
- [x] **Classical model on Raspberry Pi** (real edge demo)  
- [x] **Active learning**: Use classical confidence to query DL  
- [x] **Explainability**: Show which HOG blocks triggered SVM  
- [x] **Dataset distillation**: Train DL on classical pseudo-labels  

---

> **Key Insight**:  
> **Classical features are not obsolete — they are interpretable, lightweight, and complementary to deep learning.**

---

**Submit by Dec 1, 2025**  
**No extension beyond 3 days**  
**Plagiarism = Zero**

---

**Pro Tip**:  
> Use **classical features as a regularizer** — they prevent DL from overfitting on texture biases!  