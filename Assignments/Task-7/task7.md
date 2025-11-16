# **Assignment 7 – Submit Your Research Paper**  
---
## **Module**: Module 5 – Research & Publication in Computer Vision  
**Deadline**: **Dec 8, 2026**  
**Level**: 🔴 Advanced  
**Points**: **150 points** (Final Capstone)  
---
## **Objective**  
**Publish a complete research paper** based on your **Assignment 5 + Assignment 6** project, combining **Deep Learning**, **Classical Feature Engineering**, **Real-time Deployment**, and **Novel Contribution**.

This is your **capstone research output** — a **conference-ready paper** (IEEE, CVPR, WACV, ICCV format) that can be submitted to **peer-reviewed venues**.

---

## **Paper Requirements**

| Section | Must Include |
|--------|-------------|
| **Title** | Clear, concise, novel (≤15 words) |
| **Abstract** | 150–200 words, problem + method + results |
| **Keywords** | 5–7 (e.g., real-time detection, hybrid model, HOG-CNN fusion) |
| **Introduction** | Motivation, gap, contribution (3 bullet points) |
| **Related Work** | 8–12 citations (survey DL + classical hybrids) |
| **Methodology** | Full pipeline: Dataset → Classical → DL → Fusion → Deployment |
| **Experiments** | Dataset, metrics, ablation, real-time benchmarks |
| **Results** | Tables, graphs, confusion matrix, FPS vs accuracy |
| **Discussion** | Limitations, failure cases, insights |
| **Conclusion** | Summary + future work |
| **References** | IEEE format, ≥15 papers |
| **Appendix** | Code link, demo video, supplementary results |

---

## **Paper Structure (IEEE 2-Column Format)**

```latex
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite,amsmath,amssymb,graphicx,booktabs,caption,subcaption}
\usepackage[table,xcdraw]{xcolor}

\title{Your Paper Title Here}

\author{
\IEEEauthorblockN{Your Name}
\IEEEauthorblockA{Roll No: 22671A73XX\\
Department of CSE/AI\&ML\\
Your College Name\\
Email: your.email@domain.com}
}

\begin{document}
\maketitle

\begin{abstract}
% 150–200 words
\end{abstract}

\begin{IEEEkeywords}
real-time detection, hybrid model, HOG, YOLO, edge deployment
\end{IEEEkeywords}

\section{Introduction}
% Problem, gap, contributions

\section{Related Work}
% Cite 8–12 papers

\section{Proposed Method}
\subsection{Dataset and Preprocessing}
\subsection{Classical Feature Engineering}
\subsection{Deep Learning Backbone}
\subsection{Hybrid Fusion Strategy}
\subsection{Real-time Deployment}

\section{Experiments}
\subsection{Setup}
\subsection{Metrics}
\subsection{Ablation Study}

\section{Results and Analysis}
% Tables, Figures

\section{Discussion}

\section{Conclusion}

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
```

---

## **Novelty & Contribution (Must Be Clear)**

Your paper **must claim at least one** of the following:

| Type | Example |
|------|--------|
| **New Dataset** | "Indian Traffic Signs in Low Light" (1,000+ images) |
| **New Method** | "HOG-Guided Attention in YOLO" |
| **New Fusion** | "Dynamic Weighting of Classical + DL Confidence" |
| **New Application** | "Real-time Pothole Detection on Indian Roads" |
| **New Benchmark** | "First hybrid model achieving 42 mAP @ 35 FPS on CPU" |

---

## **Evaluation Metrics in Paper**

| Task | Metrics |
|------|--------|
| Classification | Accuracy, F1, Top-1/Top-5 |
| Detection | mAP@0.5, mAP@0.5:0.95, IoU |
| Segmentation | mIoU, Dice, Pixel Accuracy |
| **Real-time** | FPS, Latency (ms), Memory (MB), Model Size |

---

## **Results Template (Include in Paper)**

```markdown
### Table 1: Performance Comparison
| Method | mAP@0.5 | FPS (CPU) | Model Size | Accuracy |
|-------|--------|----------|------------|----------|
| YOLOv8n | 37.3 | 15 | 25 MB | - |
| HOG + SVM | 28.1 | 45 | 0.8 MB | 72% |
| **Ours (Hybrid)** | **41.8** | **32** | 26 MB | **88%** |

### Figure 3: Real-time Inference
![demo](demo_screenshot.png)
```

---

## **Deliverables**

| Item | Format | Notes |
|------|--------|-------|
| **Research Paper** | `Paper_<roll_no>.pdf` | IEEE 2-column, 6–8 pages |
| **LaTeX Source** | `paper.tex`, `references.bib` | GitHub-ready |
| **Code Repository** | GitHub Link | `README.md` with setup |
| **Demo Video** | `demo_paper.mp4` | < 3 min, narrated |
| **Poster** | `Poster_<roll_no>.pdf` | A4/A3, for presentation |
| **Presentation** | `Slides_<roll_no>.pdf` | 10 slides, 7-min talk |
| **Supplementary** | `supplementary.pdf` | Extra results, failure cases |

---

## **Submission Instructions**

1. **Fork** `learncv.ai` → `/research-papers` branch  
2. Create folder:  
   ```
   /research-papers/2025/<roll_no>/
   ```
3. Upload:  
   - `Paper_<roll_no>.pdf`  
   - `paper.tex`, `references.bib`  
   - `poster.pdf`, `slides.pdf`  
   - `demo_paper.mp4`  
   - `README.md` (with GitHub code link)  
4. Commit:  
   ```
   "Final Research Paper Submission - Roll No: 22671A73XX"
   ```
5. **Create Pull Request** to `/research-papers`

---

## **Evaluation Rubric (150 Points)**

| Criteria | Points | Description |
|--------|--------|-----------|
| **Technical Depth** | 40 | Method, math, ablation |
| **Novelty & Clarity** | 30 | Clear contribution, not incremental |
| **Experiments & Results** | 30 | Reproducible, strong baselines |
| **Writing & Structure** | 20 | IEEE format, grammar, flow |
| **Visualization** | 15 | Tables, figures, demo |
| **Presentation & Poster** | 15 | Clarity, confidence, Q&A |
| **Total** | **150** | |

---

## **Presentation Schedule (Jan 22–24, 2026)**

| Date | Time | Format |
|------|------|--------|
| Jan 22 | 10 AM – 1 PM | Poster Session |
| Jan 23 | 2 PM – 5 PM | 7-min Talks (10 slides) |
| Jan 24 | 11 AM | Best Paper Awards |

---

## **Resources**

- [IEEE LaTeX Template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhnkqqfh)  
- [CVPR 2025 Author Kit](https://cvpr.thecvf.com/)  
- [arXiv Sanity](https://arxiv.org/) – Find related work  
- [Papers with Code](https://paperswithcode.com/) – Benchmarks  
- [Zotero / Mendeley] – Reference management  

---

## **Bonus (Extra 20 Points)**

| Achievement | Points |
|------------|--------|
| Submit to **arXiv** | +10 |
| Submit to **CVPR/WACV 2026** | +20 |
| GitHub Stars ≥ 50 | +10 |
| YouTube Demo ≥ 500 views | +10 |

---

## **Final Note**

> **This is your first research paper.**  
> It doesn’t need to be perfect — it needs to be **yours**.  
> **Cite properly. Think deeply. Code cleanly. Present confidently.**

---

**Deadline**: **Dec 8, 2026, 11:59 PM IST**  
**No extensions. No excuses.**

---

**Pro Tip**:  
> **Title Formula**:  
> `"[Novelty] [Task] using [Method] for [Application] in Real-time"`  
> Example:  
> **"HOG-Guided YOLO for Real-time Pothole Detection on Indian Roads"**

---

**Ready to publish?**  
Your journey from **Assignment 1 (filters)** to **Assignment 7 (research paper)** is complete.  

**Submit. Present. Inspire.**  

--- 

**Let the world see your vision.**