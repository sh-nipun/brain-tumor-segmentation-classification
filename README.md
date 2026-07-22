# 🧠 Brain Tumor Segmentation & Grade Classification

An AI-powered web app that looks at a brain MRI scan and automatically:
1. **Finds the tumor** and outlines exactly where it is (segmentation)
2. **Classifies its grade/type** (classification)

Built as a Final Year Design Project (FYDP), combining deep learning research with a real, working Streamlit web application.

---

## 📖 What does this project actually do? (In Plain Words)

Imagine a doctor looking at an MRI scan of a patient's brain to check for a tumor. This normally takes trained expertise and time. This project trains an AI model to do two things at once, just by looking at the image:

- **Draw the exact boundary of the tumor** on the scan (like tracing its outline with a pen)
- **Tell what type/grade of tumor it might be**

Both of these are done **simultaneously by a single AI model**, called a **Multi-Task Network (MTL-Net)** — instead of using two separate models, one model learns to do both jobs together, which makes it faster and often more accurate.

The whole thing is wrapped in a simple **web app** — you upload an MRI image, and it shows you the result in seconds.

---

## ✨ Key Features

- 🖼️ Upload a brain MRI scan and get instant results
- 🎯 Tumor region highlighted directly on the image
- 🏷️ Predicted tumor grade/class shown alongside
- ⚡ Runs on GPU for fast predictions
- 🖱️ One-click launch with `run_app.bat` (no coding needed to run it)

---

## 🏆 Results

The final combined model (MTL-Net) achieved:

| Task | Metric | Score |
|---|---|---|
| Tumor Segmentation | Dice Score | **86.79%** |
| Tumor Classification | Accuracy | **97.42%** |

Along the way, several models were trained and compared to arrive at the best combination:

**Classification models tested:**
| Model | Accuracy |
|---|---|
| MobileNetV2 | 91.00% |
| EfficientNetB0 | 94.00% |
| VGG16 | 96.26% |

**Segmentation models tested:**
| Model | Dice Score |
|---|---|
| U-Net | 80.63% |
| Attention U-Net | 82.84% |
| ResU-Net | 84.98% |

The final **MTL-Net** (shared ResNet34 encoder) outperformed the individual models by learning both tasks together.

---

## 🧰 Tech Stack

- **Python**
- **PyTorch** (deep learning framework)
- **Streamlit** (web app interface)
- **BraTS 2020 Dataset** (public brain MRI dataset used for training)
- Trained and tested on an NVIDIA RTX 3060 GPU

---

## 🚀 How to Run This Project Yourself

### Option 1: Easiest way (Windows)
Just double-click:
```
run_app.bat
```
This will automatically start the app.

### Option 2: Manual way (any OS)

**Step 1 — Clone this repository**
```bash
git clone https://github.com/sh-nipun/brain-tumor-segmentation-classification.git
cd brain-tumor-segmentation-classification
```

**Step 2 — Install the required packages**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the app**
```bash
streamlit run app.py
```

**Step 4 — Open your browser**
Streamlit will automatically open a browser tab (usually at `http://localhost:8501`). Upload an MRI image and see the results!

> ⚠️ **Note:** The trained model file (`multitask_best.pth`, ~95 MB) is stored using **Git LFS**. Make sure you have [Git LFS](https://git-lfs.com) installed before cloning, otherwise the model file may not download correctly:
> ```bash
> git lfs install
> ```

---

## 📁 Project Structure

```
brain_tumor_app/
├── app.py                 # Main Streamlit web application
├── multitask_best.pth     # Trained MTL-Net model weights (via Git LFS)
├── requirements.txt       # Python dependencies
├── run_app.bat            # One-click launcher (Windows)
└── README.md
```

---

## 🎓 About This Project

This project was developed as part of a **B.Sc. Final Year Design Project (FYDP)** titled:

> *"An Efficient Multi-Task Network for Brain Tumor Segmentation and Grade Classification on MRI Data"*

It combines academic research (comparing multiple deep learning architectures) with a practical, deployable tool — bridging the gap between a research notebook and something an actual user can interact with.

---

## ⚠️ Disclaimer

This project is for **academic and research purposes only**. It is **not** a certified medical diagnostic tool and should **not** be used for real clinical decisions. Always consult a qualified medical professional for diagnosis and treatment.

---

## 📬 Contact

Made by **Nipun**
GitHub: [@sh-nipun](https://github.com/sh-nipun)
