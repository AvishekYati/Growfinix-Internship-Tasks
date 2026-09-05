# Task 1: AI-Powered Automated Media Metadata Extraction Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11s-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end computer vision and image processing pipeline designed to ingest images and video footage, run automated object detection, extract dominant color palettes, analyze optical properties (lighting, exposure, contrast, sharpness, and noise), classify scene semantics, score cinematic quality, and generate structured JSON metadata and summary audit reports.

Developed as part of the **Growfinix Technology Data Science Internship (Month 3: Advanced AI, Deep Learning & LLMs)**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Analytical Modules](#-analytical-modules)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Execution & Usage](#-execution--usage)
- [Sample Outputs & Reports](#-sample-outputs--reports)
- [Limitations & Future Improvements](#-limitations--future-improvements)

---

## 📖 Overview

Manual media tagging and quality inspection across large digital asset management (DAM) libraries is time-consuming and prone to human inconsistency. 

This project implements an automated multi-stage analysis pipeline that computes:
1. **Physical & Technical Metadata:** Resolution, aspect ratio, encoding format, file size, frame rate, and duration.
2. **Deep Learning Object Detection:** Pre-trained YOLO11s detector with class tagging, confidence scoring, and bounding box area ratios.
3. **Colorimetry & Dominant Palette Extraction:** K-Means clustering in RGB space, HSV color classification, color temperature estimation, and saturation level.
4. **Photometric & Quality Diagnostics:** Mean/median brightness, shadow/highlight pixel distribution, exposure classification, RMS contrast, Laplacian variance (blur detection), and median-filtered residual noise.
5. **Heuristic Scene & Cinematic Scoring:** Indoor/outdoor classification, category tagging, and a weighted multi-factor cinematic index.

---

## ⚡ Key Features

* **Dual-Format Processing:** Full support for single images (`.jpg`, `.png`, `.webp`, etc.) and video clips (`.mp4`, `.avi`, `.mov`).
* **YOLO11s Deep Learning Integration:** Real-time object identification with configurable confidence (`0.40`) and IoU (`0.50`) thresholds.
* **K-Means Color Palette:** Extracts top $k$ dominant colors, maps them to HEX, RGB, and human-readable names, and exports visual palette bar charts.
* **Temporal Video Analysis:** Evaluates sampled frames across configurable intervals, detects cut/scene transitions via frame-difference metrics, and outputs fully annotated MP4 videos.
* **Automated Tagging & Reporting:** Produces JSON metadata for every asset and consolidates entire folders into tabular `media_inventory.csv` and `video_inventory.csv`.

---

## 🏗 System Architecture

```text
Raw Media Input (Images / Videos)
               │
               ▼
┌──────────────────────────────────────────────┐
│        File & Technical Extraction           │
│ (Dimensions, Resolution, Aspect Ratio, Codec)│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│           YOLO11s Object Detection           │
│   (Bounding Boxes, Labels, Confidence)       │
└──────────────────────┬───────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│ Colorimetric Analysis   │ │ Photometric & Quality   │
│ - K-Means Clustering    │ │ - Laplacian Blur Score  │
│ - HSV / Temperature     │ │ - Contrast / Dynamic Rng│
│ - Dominant Palette (HEX)│ │ - Brightness & Exposure │
└──────────────┬──────────┘ └─────────────┬───────────┘
               │                          │
               └──────────────┬───────────┘
                              │
                              ▼
┌──────────────────────────────────────────────┐
│        Scene Classification & Heuristics     │
│   (Indoor/Outdoor, Cinematic Rating, Tags)   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│               Artifact Output                │
│ - JSON Metadata     - Annotated Media        │
│ - Color Swatches    - Inventory CSV Reports  │
└──────────────────────────────────────────────┘
