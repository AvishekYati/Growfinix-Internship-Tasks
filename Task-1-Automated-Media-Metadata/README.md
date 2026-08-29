# Task 1 - Automated Image/Video Metadata Extraction

## Growfinix Technology Data Science Internship

### Month 3 - Advanced AI, Deep Learning and LLMs

---

## 1. Project Overview

This project implements an automated computer vision pipeline for analyzing digital images and video frames.

The system is designed to automatically extract useful metadata from visual media including object information, color information, lighting characteristics, contrast measurements and other image/video properties.

The project combines computer vision, deep learning and image processing techniques to convert unstructured visual media into structured metadata.

---

## 2. Problem Statement

Digital media collections can contain thousands of images and videos that require manual organization and tagging.

Manually identifying objects, analyzing colors, evaluating lighting and generating metadata is time-consuming.

This project automates these processes using computer vision and deep learning.

The system analyzes media assets and generates structured information that can be used for:

- Digital asset management
- Media organization
- Automatic tagging
- Image search
- Video indexing
- Content analysis
- Computer vision applications

---

## 3. Objectives

The main objectives are:

1. Automatically analyze images.
2. Process video frames.
3. Detect objects using a deep learning object detection model.
4. Extract dominant colors.
5. Generate RGB and HEX color information.
6. Analyze image lighting.
7. Calculate image contrast.
8. Evaluate image quality characteristics.
9. Generate automatic metadata.
10. Produce annotated visual outputs.
11. Store analysis results in structured formats.

---

## 4. Features

### Object Detection

The system detects objects present in an image or video frame.

For detected objects, the system can provide:

- Object class
- Confidence score
- Bounding box coordinates

---

### Color Analysis

The pipeline analyzes the visual color composition of the media.

It can extract:

- Dominant colors
- RGB values
- HEX values
- Color proportions
- Color palette visualization

---

### Lighting Analysis

Lighting characteristics are analyzed using image intensity information.

The system evaluates:

- Average brightness
- Brightness distribution
- Underexposure
- Overexposure
- Lighting condition

---

### Contrast Analysis

The system evaluates the contrast of the image using pixel intensity statistics.

This helps classify images based on their visual dynamic range.

---

### Image Quality Analysis

Additional visual properties can be analyzed, including:

- Sharpness
- Blur
- Brightness
- Contrast
- Resolution
- Image dimensions

---

### Video Analysis

Videos are processed frame-by-frame.

The pipeline can:

1. Open the video.
2. Read video frames.
3. Sample frames.
4. Detect objects.
5. Analyze colors.
6. Analyze lighting.
7. Analyze contrast.
8. Generate metadata.
9. Produce annotated video/output.

---

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image and video processing |
| PyTorch | Deep learning framework |
| YOLO | Object detection |
| NumPy | Numerical processing |
| Pandas | Data processing |
| Matplotlib | Visualization |
| Scikit-learn | Machine learning utilities |
| Pillow | Image processing |
| Jupyter Notebook | Development and demonstration |

---

## 6. System Architecture

```text
              INPUT MEDIA
                  |
          +-------+-------+
          |               |
       IMAGE           VIDEO
          |               |
          |        Frame Extraction
          |               |
          +-------+-------+
                  |
          Image Preprocessing
                  |
          +-------+-------+
          |               |
          |        Object Detection
          |               |
          |        Color Analysis
          |               |
          |        Lighting Analysis
          |               |
          |        Contrast Analysis
          |               |
          |        Quality Analysis
          |               |
          +-------+-------+
                  |
          Metadata Generation
                  |
          +-------+-------+
          |       |       |
       JSON     CSV    Visualizations
          |       |       |
          +-------+-------+
                  |
             FINAL OUTPUT
