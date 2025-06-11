# 🚦 Smart Vehicle Speed Estimation (NoDL & NoYOLO) 🚦

A lightweight, intelligent system to detect vehicles and estimate their speed from video — **without any Deep Learning or YOLO!**  
Utilizing classical Computer Vision techniques for efficient, real-time performance on regular hardware.

---

## 🔥 Why Choose This?

- **No Deep Learning (NoDL)**  
  No need for massive datasets, model training, or heavy frameworks.  
- **No YOLO!**  
  Avoids complex neural network detectors; relies on traditional, well-proven CV methods.  
- **Smart & Efficient**  
  Combines background subtraction, centroid tracking, perspective transform, and ORB feature matching.  
- **Real-Time Friendly**  
  Runs smoothly on standard PCs or edge devices with limited resources.

---

## 🚗 Key Features

- Vehicle detection with **Background Subtractor MOG2**  
- Reliable object tracking via a custom **Centroid Tracker**  
- Dynamic ROI tracking using **ORB feature matching** for perspective correction  
- Speed estimation based on entry-exit crossing time and real-world distance  
- Bird’s-eye view visualization for easy interpretation  
- Automatic CSV output logging vehicle speeds and classifications

---

## IMAGES
![bird_view_100](https://github.com/user-attachments/assets/55a9d2ea-4e1f-48a4-83f6-b9245a93a6dc)
![Screenshot 2025-06-11 101116](https://github.com/user-attachments/assets/3e44de87-e859-4233-9a52-3e7bdfb228a5)

---

## ⚙️ How to Use

1. Ensure you have the ROI calibration file (`saved_roi1.npy`) ready (from manual annotation).  
2. Run the main script:

```bash
python birdeyes.py

