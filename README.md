# HECKTOR2025 – Team MEDAI  
Solutions for the three tasks of the **HECKTOR 2025 Challenge**.  

[👉 HECKTOR 2025 Challenge Website](https://hecktor25.grand-challenge.org/)  

---

## Overview  
This repository contains our approaches for:  
1. **Task 1 – GTVp and GTVn Segmentation**  
2. **Task 2 – Recurrence-Free Survival Prediction**  
3. **Task 3 – HPV Status Prediction**  

For official data, task descriptions, and challenge details, please refer to the [HECKTOR 2025 challenge page](https://hecktor25.grand-challenge.org/).  

---

## Data Preprocessing  
See: [`preprocessing.py`](preprocessing.py)  

- All images were **resampled** to **1 × 1 × 1 mm³**  
- **Overlapping bounding boxes** between CT and PET volumes were computed  
- A **region of interest (ROI)** was automatically defined using PET intensities  
- The **top portion** of the PET scan was analyzed to locate the **largest high-intensity region**  
- The **crop center** was set at the **centroid** of this region  
- A **fixed-size crop (200 × 200 × 310)** was extracted for CT, PET, and masks  

---

## Methods  

### Task 1 – GTVp & GTVn Segmentation  
We used **STU-Net (Small)** – see the official implementation [here](https://github.com/uni-medical/STU-Net/tree/main/nnUNet-2.2).  

### Task 2 – Recurrence-Free Survival Prediction  
*Details coming soon*  

### Task 3 – HPV Status Prediction  
*Details coming soon*  

---

## Docker Containers  
We provide pre-built Docker images for reproducibility.  
If you’d like access, please reach out by email.  

---


