# 🎯 Computer Vision Bootcamp
**A 4-Day Intensive Journey from Foundations to Deployment**

Made with ♡ by Aryan

---

## 📚 Overview

This bootcamp provides a comprehensive, hands-on introduction to Computer Vision (CV) using Python. From fundamental NumPy operations to deploying production-ready models, you'll gain practical skills through interactive Jupyter notebooks, real code examples, and mini-projects.

**Target Audience:** Beginners and intermediate learners who want to build a strong foundation in Computer Vision and Deep Learning.

---

## 🎓 What You'll Learn

### Day 1: CV Foundations
- **NumPy Essentials:** Arrays, broadcasting, vectorization (50-100x speedup!)
- **Pandas for Vision:** Dataset management, train/val/test splitting
- **OpenCV Basics:** Image I/O, color spaces, transformations, edge detection

### Day 2: Preprocessing & Classical CV
- **Normalization:** Min-Max scaling, Z-score standardization, ImageNet stats
- **Data Augmentation:** Geometric & color transformations to expand training data
- **Classical Techniques:** Edge detection, contours, ORB features, template matching, Haar cascades

### Day 3: Deep Learning & CNNs
- **CNN Building Blocks:** Convolutional layers, activations, pooling
- **Training CNNs:** Complete CIFAR-10 training pipeline with PyTorch
- **Transfer Learning:** Fine-tuning pretrained models (ResNet, VGG, MobileNet)

### Day 4: Model Deployment
- **Model Export:** ONNX and TorchScript for cross-platform deployment
- **FastAPI Backend:** Building REST APIs for model inference
- **Streamlit UI:** Creating interactive web interfaces

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Basic Python programming knowledge
- 4-8 GB RAM recommended

### Installation

1. **Clone or download this repository:**
```bash
cd CV-sessions
```

2. **Create a virtual environment (recommended):**
```bash
# Using venv
python -m venv cv_env

# Activate on Linux/Mac
source cv_env/bin/activate

# Activate on Windows
cv_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

5. **Open Day1/01_NumPy_Essentials.ipynb and start learning!**

---

## 📂 Repository Structure

```
CV-sessions/
├── Day1/                           # Foundations
│   ├── 01_NumPy_Essentials.ipynb
│   ├── 02_Pandas_Vision_Datasets.ipynb
│   ├── 03_OpenCV_Basics.ipynb
│   ├── Day1_CV_Foundations.pdf
│   ├── Day1_CV_Foundations.tex
│   ├── Day1_Assignment_Question.pdf
│   └── Day1_Assignment_Question.tex
│
├── Day2/                           # Preprocessing & Classical CV
│   ├── 01_Normalization_Standardization.ipynb
│   ├── 02_Data_Augmentation.ipynb
│   ├── 03_Classical_CV_Techniques.ipynb
│   ├── Day2_Preprocessing_Classical_CV.pdf
│   ├── Day2_Preprocessing_Classical_CV.tex
│   ├── Day2_Assignment_Question.pdf
│   └── Day2_Assignment_Question.tex
│
├── Day3/                           # Deep Learning & CNNs
│   ├── 01_CNN_Building_Blocks.ipynb
│   ├── 02_Training_CNNs.ipynb
│   ├── 03_Transfer_Learning.ipynb
│   ├── Day3_Deep_Learning_CNNs.pdf
│   ├── Day3_Deep_Learning_CNNs.tex
│   ├── Day3_Assignment_Question.pdf
│   └── Day3_Assignment_Question.tex
│
├── Day4/                           # Model Deployment
│   ├── 01_Model_Export.ipynb
│   ├── 02_FastAPI_Backend.ipynb
│   ├── 03_Streamlit_UI.ipynb
│   ├── Day4_Deployment_Pipeline.pdf
│   ├── Day4_Deployment_Pipeline.tex
│   ├── Day4_Final_Project_Options.pdf
│   └── Day4_Final_Project_Options.tex
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 📖 Day-by-Day Breakdown

### **Day 1: Computer Vision Foundations**
**Duration:** 3-4 hours | **Difficulty:** ⭐ Beginner

#### Notebooks:
1. **01_NumPy_Essentials.ipynb** (53 cells)
   - Array creation and manipulation
   - Indexing, slicing, and fancy indexing
   - Broadcasting for efficient operations
   - Vectorization vs loops (demo: 62.7x speedup!)
   - Image-specific operations

2. **02_Pandas_Vision_Datasets.ipynb** (35 cells)
   - Creating DataFrames for image datasets
   - Filtering, grouping, and aggregation
   - Train/validation/test splitting with stratification
   - Handling missing values and duplicates
   - Merging annotations with image paths

3. **03_OpenCV_Basics.ipynb** (30 cells)
   - Reading/writing images (cv2.imread, cv2.imwrite)
   - Color space conversions (RGB, BGR, Grayscale, HSV)
   - Resizing and cropping with aspect ratio preservation
   - Blurring techniques (Gaussian, Median, Bilateral)
   - Canny edge detection

#### Learning Outcomes:
✅ Understand image representation as NumPy arrays  
✅ Manage vision datasets with Pandas  
✅ Perform basic image processing with OpenCV  
✅ Complete edge detection mini-project

---

### **Day 2: Preprocessing & Classical CV**
**Duration:** 4-5 hours | **Difficulty:** ⭐⭐ Intermediate

#### Notebooks:
1. **01_Normalization_Standardization.ipynb** (20 cells)
   - Min-Max normalization to [0,1] and [-1,1]
   - Z-score standardization (mean=0, std=1)
   - ImageNet normalization (mean=[0.485, 0.456, 0.406])
   - Per-channel vs global normalization
   - Visualizing normalized distributions

2. **02_Data_Augmentation.ipynb** (25 cells)
   - Geometric transforms: flip, rotate, scale, translate, shear
   - Color augmentations: brightness, contrast, saturation, hue
   - Adding noise (Gaussian, Salt & Pepper)
   - Creating augmentation pipelines
   - Before/after comparisons

3. **03_Classical_CV_Techniques.ipynb** (30 cells)
   - Edge detection: Sobel, Laplacian, Canny
   - Contour detection and analysis
   - Hough transforms (lines, circles)
   - ORB feature detection and matching
   - Template matching
   - Haar cascade face detection

#### Learning Outcomes:
✅ Normalize images for neural networks  
✅ Augment datasets to improve model generalization  
✅ Apply classical CV algorithms still relevant today  
✅ Understand feature detection before deep learning

---

### **Day 3: Deep Learning & CNNs**
**Duration:** 5-6 hours | **Difficulty:** ⭐⭐⭐ Advanced

#### Notebooks:
1. **01_CNN_Building_Blocks.ipynb** (12 cells)
   - Convolutional layers (nn.Conv2d)
   - Activation functions (ReLU, Sigmoid, Tanh, LeakyReLU)
   - Pooling layers (MaxPool2d, AvgPool2d)
   - Building a complete SimpleCNN from scratch
   - Understanding feature map dimensions

2. **02_Training_CNNs.ipynb** (10 cells)
   - CIFAR-10 dataset loading with transforms
   - Defining CNN architecture
   - Training loop with backpropagation
   - Validation and testing
   - Loss/accuracy plotting and visualization

3. **03_Transfer_Learning.ipynb** (11 cells)
   - Loading pretrained ResNet18 from torchvision
   - Freezing vs unfreezing layers
   - Replacing final classification layer
   - Fine-tuning strategies
   - Comparing ResNet/VGG/MobileNet architectures

#### Learning Outcomes:
✅ Understand CNN components and how they work  
✅ Train a CNN from scratch on CIFAR-10  
✅ Leverage pretrained models with transfer learning  
✅ Know when to freeze/unfreeze layers

---

### **Day 4: Model Deployment**
**Duration:** 4-5 hours | **Difficulty:** ⭐⭐⭐ Advanced

#### Notebooks:
1. **01_Model_Export.ipynb** (16 cells)
   - Exporting PyTorch models to ONNX format
   - ONNX Runtime inference
   - TorchScript (tracing vs scripting)
   - Model verification and validation
   - Cross-platform deployment considerations

2. **02_FastAPI_Backend.ipynb** (12 cells)
   - FastAPI server structure
   - Creating prediction endpoints
   - File upload handling
   - Request/response models with Pydantic
   - Error handling and logging
   - Docker deployment

3. **03_Streamlit_UI.ipynb** (14 cells)
   - Streamlit components (widgets, layouts)
   - Image upload and display
   - Real-time prediction UI
   - Multi-tab layouts
   - Custom styling with CSS
   - Deployment to Streamlit Cloud

#### Learning Outcomes:
✅ Export models in production-ready formats  
✅ Build REST APIs with FastAPI  
✅ Create interactive web UIs with Streamlit  
✅ Deploy models to the cloud

---

## 💻 Running the Notebooks

### Option 1: Jupyter Notebook (Recommended for Beginners)
```bash
jupyter notebook
# Navigate to Day1/01_NumPy_Essentials.ipynb
```

### Option 2: JupyterLab (Modern Interface)
```bash
pip install jupyterlab
jupyter lab
```

### Option 3: VS Code
1. Install the "Jupyter" extension
2. Open any `.ipynb` file
3. Select Python kernel from virtual environment

---

## 🔧 Troubleshooting

### Import Errors
```bash
# If you get "ModuleNotFoundError"
pip install --upgrade -r requirements.txt
```

### OpenCV Display Issues
```python
# If cv2.imshow() doesn't work in notebooks, use matplotlib instead:
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

### PyTorch Installation
```bash
# For CPU-only (faster install, smaller size):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8 (if you have NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
If you run into memory errors when training models:
- Reduce batch size in training notebooks
- Close other applications
- Use smaller model architectures
- Work with subset of dataset

---

## 📝 Assignments

Each day includes:
- **Mini-tasks** embedded in notebooks
- **End-of-day assignment** (PDF in each Day folder)
- Recommended 1-2 hours per assignment

**Day 1:** Pencil Sketch Effect  
**Day 2:** Build Complete Augmentation Pipeline  
**Day 3:** Fine-tune Model on Custom Dataset  
**Day 4:** Deploy Full-Stack CV Application

---

## 🎯 Final Project Options

After completing all 4 days, choose one:

1. **Object Detection System** - YOLOv8 with custom dataset
2. **Image Classification App** - End-to-end deployment
3. **Face Recognition System** - Real-time webcam detection
4. **OCR Application** - Document text extraction
5. **Custom Project** - Propose your own!

Detailed requirements in `Day4/Day4_Final_Project_Options.pdf`

---

## 📚 Additional Resources

### Documentation
- [NumPy Docs](https://numpy.org/doc/stable/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [OpenCV Docs](https://docs.opencv.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)

### Further Learning
- **Books:**
  - "Deep Learning for Computer Vision" by Rajalingappaa Shanmugamani
  - "Hands-On Computer Vision with TensorFlow 2" by Eliot Andres
  
- **Online Courses:**
  - Fast.ai Practical Deep Learning for Coders
  - Stanford CS231n (Convolutional Neural Networks)
  
- **Datasets:**
  - [Kaggle Datasets](https://www.kaggle.com/datasets)
  - [Roboflow Universe](https://universe.roboflow.com/)
  - [Papers with Code](https://paperswithcode.com/datasets)

---

## 🤝 Contributing

Found an issue or want to improve the content?
1. Report bugs or suggest features
2. Submit corrections to notebooks
3. Share your project implementations

---

## 📜 License

This educational material is provided for learning purposes. Feel free to use and modify for personal and educational use.

---

## 👨‍🏫 About the Instructor

**Aryan** - Computer Vision Engineer & Educator

Connect for questions, feedback, or collaboration opportunities!

---

## 🌟 Acknowledgments

Special thanks to:
- The open-source community (NumPy, PyTorch, OpenCV teams)
- All students who provided feedback
- Contributors to this bootcamp

---

## 📊 Progress Checklist

Track your learning journey:

- [ ] **Day 1 Complete**
  - [ ] NumPy Essentials
  - [ ] Pandas Vision Datasets
  - [ ] OpenCV Basics
  - [ ] Day 1 Assignment

- [ ] **Day 2 Complete**
  - [ ] Normalization & Standardization
  - [ ] Data Augmentation
  - [ ] Classical CV Techniques
  - [ ] Day 2 Assignment

- [ ] **Day 3 Complete**
  - [ ] CNN Building Blocks
  - [ ] Training CNNs
  - [ ] Transfer Learning
  - [ ] Day 3 Assignment

- [ ] **Day 4 Complete**
  - [ ] Model Export
  - [ ] FastAPI Backend
  - [ ] Streamlit UI
  - [ ] Day 4 Assignment

- [ ] **Final Project**

---

## 💡 Tips for Success

1. **Run Every Code Cell** - Don't just read, execute and experiment!
2. **Modify Parameters** - Change values to see what happens
3. **Take Notes** - Write comments in code cells
4. **Ask Questions** - No question is too simple
5. **Build Projects** - Apply knowledge to your own ideas
6. **Join Community** - Share your progress and help others

---

**Happy Learning! 🚀**

*Last Updated: December 2024*
