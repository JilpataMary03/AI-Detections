# Real-Time Person Collision Detection using YOLOv8

## 📌 Overview
This project detects **real human-to-human collisions** in real-time video streams using:

- YOLOv8 for person detection
- A custom lightweight multi-object tracker
- Motion-based collision verification logic

Unlike simple bounding-box overlap methods, this system uses **motion reasoning** to reduce false positives.

---

## 🎯 Key Features
- 🧍 **Person Detection using YOLOv8**
- 🧠 **Custom Object Tracking**
  - IOU-based association
  - Track history management
- ⚡ **Collision Event Detection**
  - Approaching motion detection
  - Sudden speed drop analysis
  - Stationary person filtering
- 🎥 **RTSP Stream Support**
- 🚀 **Optimized for Real-Time Execution**
  - Frame skipping
  - ONNX inference

---

## 🧠 Collision Detection Logic

A collision is confirmed only when:
1. Two persons move **towards each other**
2. Bounding boxes become **very close / overlap**
3. At least one person shows a **sudden speed drop**
4. The event persists across multiple frames

This multi-stage logic significantly reduces false alerts.

---

## 🛠️ Tech Stack
- Python
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy
- ONNX Runtime

---

## ▶️ How to Run

```bash
pip install -r requirements.txt

