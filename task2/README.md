# Sentinel-2 image matching

### Overview
This project focuses on developing an algorithm for matching satellite images using feature detection and description techniques. The initial dataset of Sentinel-2 images was preprocessed by extracting 1024×1024 fragments while discarding partially black or incomplete ones. The approach allows visualization and comparison of keypoints between images captured in different seasons. ORB was used as the feature descriptor, providing a balance between computational efficiency and accuracy, though it shows sensitivity to brightness variations, especially in cloudy regions.


---

## Setup Instructions

### 1. Create a Virtual Environment

Navigate to the project folder and run:

```bash
python -m venv myenv
```

### 2. Activate the Virtual Environment

**For Linux/macOS:**

```bash
source myenv/bin/activate
```

**For Windows (CMD):**

```bash
myenv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running algorithm


```bash
python src/train.py --folder data/2018 
```

---

## Inference


```bash
python src/inference.py --features features.pkl --img1 data/2018/T36UXA_20180904T083549_TCI.png
```

---

## Running the Notebook

1. Install the environment kernel:

   ```bash
   python -m ipykernel install --user --name=.myvenv --display-name "demo-env"
   jupyter notebook
   ```
2. Open the notebook file `demo.ipynb`.
3. Select the kernel **demo-env** and run all cells.

---
## Download images

Download the images (folder data/) from the provided link and place them in the root directory (task2/) before running the project.


**[Download images](https://drive.google.com/drive/folders/1LaRpOROiHCd6lKG9F_dDF6qlnrYN5Lk5?usp=sharing)**

---
## Possible Improvements

The project can be further improved by using advanced feature detectors and descriptors such as SIFT, AKAZE. Matching accuracy can be enhanced through better algorithms like FLANN, ratio test, RANSAC to reduce false correspondences. Additionally, machine learning models such as CNNs or Transformers can be applied to learn robust feature matching directly from data.
