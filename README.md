# Cephalometry Using YOLOv9

A computer vision project that applies YOLOv9 object detection for automated cephalometric analysis in dental and orthodontic imaging. This project enables precise detection and measurement of anatomical landmarks in cephalometric X-ray images.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Configuration](#model-configuration)
- [Detection Parameters](#detection-parameters)
- [Output Formats](#output-formats)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

Cephalometry is a diagnostic tool used in orthodontics and oral surgery to analyze the relationships between dental and skeletal structures. This project leverages the power of YOLOv9, a state-of-the-art object detection model, to automatically identify and locate anatomical landmarks in cephalometric radiographs.

### Key Benefits:
- **Automated Analysis**: Reduces manual measurement time and human error
- **High Precision**: YOLOv9's advanced architecture ensures accurate landmark detection
- **Clinical Integration**: Designed for integration into dental practice workflows
- **Flexible Output**: Multiple output formats for different use cases

## ✨ Features

- **Real-time Detection**: Fast inference on cephalometric images
- **Multiple Input Sources**: Support for images, videos, webcam, and URLs
- **Customizable Confidence Thresholds**: Adjustable detection sensitivity
- **Multiple Output Formats**: 
  - Annotated images with bounding boxes
  - Text files with coordinates
  - Cropped landmark regions
- **GPU Acceleration**: CUDA support for faster processing
- **Batch Processing**: Process multiple images simultaneously

## 🛠️ Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster inference)
- Minimum 8GB RAM
- OpenCV-compatible system

### Core Dependencies

```txt
torch>=1.8.0
torchvision>=0.9.0
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.0.0
PyYAML>=5.4.0
tqdm>=4.60.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.3.0
pathlib
argparse
```

### YOLOv9 Specific Libraries
```txt
ultralytics
thop
tensorboard
protobuf<4.21.3
```

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/selvatharrun/cephalometry-using-yolov9-.git
cd cephalometry-using-yolov9-
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv cephalo_env
source cephalo_env/bin/activate  # On Windows: cephalo_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA Support (if available)
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio
```

### 5. Download Pre-trained Models
```bash
# Download YOLOv9 weights (replace with your trained cephalometry model)
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9c.pt
```

## 📁 Project Structure

```
cephalometry-using-yolov9-/
├── detect.py                 # Main detection script
├── models/
│   ├── common.py            # Model architecture components
│   └── ...
├── utils/
│   ├── dataloaders.py       # Data loading utilities
│   ├── general.py           # General utility functions
│   ├── plots.py             # Visualization utilities
│   └── torch_utils.py       # PyTorch utilities
├── data/
│   ├── images/              # Input images directory
│   └── coco.yaml            # Dataset configuration
├── runs/
│   └── detect/              # Output directory
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Usage

### Basic Detection
```bash
python detect.py --source data/images --weights yolo.pt
```

### Advanced Usage Examples

#### Process Single Image
```bash
python detect.py --source path/to/image.jpg --weights yolo.pt --conf-thres 0.5
```

#### Process Video
```bash
python detect.py --source path/to/video.mp4 --weights yolo.pt --save-txt
```

#### Webcam Detection
```bash
python detect.py --source 0 --weights yolo.pt --view-img
```

#### Batch Processing with Custom Output
```bash
python detect.py \
    --source data/images \
    --weights yolo.pt \
    --project runs/cephalometry \
    --name experiment1 \
    --save-txt \
    --save-conf \
    --conf-thres 0.6 \
    --iou-thres 0.4
```

## ⚙️ Model Configuration

### Dataset Configuration (data/coco.yaml)
```yaml
# Modify for cephalometric landmarks
names:
  0: nasion
  1: sella
  2: orbitale
  3: porion
  4: anterior_nasal_spine
  5: posterior_nasal_spine
  6: pogonion
  7: menton
  8: gonion
  9: articulare
```

### Detection Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `--conf-thres` | Confidence threshold | 0.25 | 0.3-0.7 |
| `--iou-thres` | IoU threshold for NMS | 0.45 | 0.4-0.6 |
| `--imgsz` | Input image size | 640 | 640-1280 |
| `--max-det` | Maximum detections | 1000 | 50-200 |
| `--line-thickness` | Bounding box thickness | 1 | 1-5 |

## 📊 Output Formats

### 1. Annotated Images
- Images with bounding boxes around detected landmarks
- Confidence scores displayed
- Saved in `runs/detect/exp/`

### 2. Text Files (--save-txt)
Format: `class x_center y_center width height confidence`
```
0 0.5234 0.3456 0.0234 0.0345 0.89
1 0.6123 0.4567 0.0198 0.0276 0.92
```

### 3. Cropped Images (--save-crop)
- Individual landmark regions saved as separate images
- Useful for detailed analysis

## 🔧 Customization

### Training Your Own Model
1. Prepare cephalometric dataset with landmark annotations
2. Configure dataset YAML file
3. Train using YOLOv9 training script
4. Replace `yolo.pt` with your trained weights

### Adjusting for Clinical Use
- Modify confidence thresholds based on clinical requirements
- Customize landmark classes in configuration
- Implement post-processing for measurements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv9 team for the excellent object detection framework
- OpenCV community for computer vision tools
- PyTorch team for the deep learning framework


---

*For more detailed information about YOLOv9 architecture and training procedures, refer to the original YOLOv9 documentation.*
