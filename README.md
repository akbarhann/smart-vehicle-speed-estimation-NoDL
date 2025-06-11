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

## ⚙️ How to Use

1. Ensure you have the ROI calibration file (`saved_roi1.npy`) ready (from manual annotation).  
2. Run the main script:

```bash
python ishospeed.py

