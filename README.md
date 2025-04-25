## **Student-FaceVault**

## Overview

**Student-FaceVault** is a web application for creating and managing face recognition galleries, primarily designed for educational institutions to organize student face data by batch/year and department. The system processes videos to extract faces, builds recognition galleries, and provides a user-friendly interface for face recognition.

## Features

- **Video Processing**: Extract faces from videos with automatic face detection
- **Gallery Creation**: Create and update face recognition galleries by batch year and department
- **Face Recognition**: Upload images to identify faces using selected galleries
- **Administrative Tools**: Manage batch years and departments through the admin panel
- **User-Friendly Interface**: Intuitive web interface built with Bootstrap and modern JavaScript

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Face Detection**: YOLOv8
- **Face Recognition**: LightCNN
- **Data Storage**: File system for face data and galleries, SQLite for configuration

## Project Structure

```
Face_data-application/
├── src/                  # Backend Python code
│   ├── main.py           # FastAPI application entry point
│   ├── gallery_manager.py # Face gallery creation and management
│   ├── database.py       # Database operations
│   ├── preprocess_images.py # Image preprocessing utilities
│   └── yolo/             # YOLO face detection implementation
├── static/               # Frontend assets
│   ├── index.html        # Single page application
│   ├── css/              # CSS stylesheets
│   ├── js/               # JavaScript files
│   └── img/              # Images and icons
├── gallery/              # Face recognition data
│   ├── data/             # Processed face data
│   └── galleries/        # Generated face galleries
├── vid/                  # Input videos directory (optional)
```

## Installation and Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- Node.js and npm (for frontend development)

### Installation

1. **Clone the repository**:

   ```
   git clone https://github.com/your-username/Face_data-application.git
   cd Face_data-application
   ```
2. **Create and activate a virtual environment**:

   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```
4. **Download model weights**:

   - Download LightCNN model weights and place in `src/checkpoints/`
   - Download YOLO face detection model and place in `src/yolo/weights/`
5. **Initialize the database**:

   ```
   python src/database.py
   ```

### Running the Application

1. **Start the server**:

   ```
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```
2. **Access the application**:
   Open a web browser and navigate to `http://localhost:8000`

## Usage Guide

### 1. Process Videos

1. Navigate to the "Process Videos" section
2. Select batch year and department
3. Enter the path to videos directory
4. Click "Process Video Frames"
5. Wait for processing to complete

### 2. Create Galleries

1. Navigate to the "Create Galleries" section
2. Select batch year and department
3. Optionally check "Update existing gallery"
4. Click "Create/Update Gallery"

### 3. View Galleries

1. Navigate to the "View Galleries" section
2. Browse available galleries by department and year
3. Click on a gallery to view details and statistics

### 4. Perform Face Recognition

1. Navigate to the "Face Recognition" section
2. Upload an image containing faces
3. Select one or more galleries to search
4. Click "Recognize Faces"
5. View results showing identified persons with confidence scores

### 5. Administrative Tasks

1. Navigate to the "Admin" section
2. Manage batch years (add/delete)
3. Manage departments (add/delete)

## Technical Details

### Video Processing Pipeline

1. Extract frames from videos at regular intervals
2. Detect faces using YOLOv8 face detection model
3. Crop and preprocess detected faces
4. Save processed faces to organized directories by identity

### Gallery Creation Process

1. Extract embeddings for each face using LightCNN
2. Create average embeddings for each identity
3. Store embeddings in a gallery file (.pth)
4. Optionally augment faces to improve recognition accuracy

### Face Recognition Algorithm

1. Detect faces in the input image
2. Extract embeddings for each detected face
3. Compare embeddings with galleries using cosine similarity
4. Apply identity assignment with no-duplicate rule
5. Return results with confidence scores

## Acknowledgments

- YOLOv8 by Ultralytics
- LightCNN by AlfredXiangWu
