# Urban Development Analysis and Prediction

> A cloud-aware spatio-temporal framework for analyzing historical urban expansion and forecasting future building development from multi-temporal satellite imagery.

## Overview

Urban areas evolve continuously, making timely monitoring and forecasting important for urban planning, infrastructure development, and environmental management. Traditional satellite-image analysis often relies on manual interpretation, while many deep learning approaches focus on individual images without explicitly modelling how urban development changes over time.

This project presents an end-to-end system for **historical urban growth analysis and future urban expansion prediction** using multi-temporal satellite imagery from the **SpaceNet-7 dataset**.

The framework combines:

- **K-Means clustering and spatial analysis** for historical urban-growth analysis
- **CNNs** for spatial feature extraction
- **Transformers** for long-range temporal dependency modelling
- **LSTMs** for sequential urban-growth modelling
- **U-Net with a ResNet-50 encoder** for cloud segmentation
- **Auto-regressive forecasting** for multi-step future prediction
- **FastAPI** backend for model inference and analysis
- **HTML/CSS/JavaScript** frontend for interactive visualization

The proposed cloud-aware hybrid model achieved an **IoU of 0.62**, with a **Dice score of 0.74, Precision of 0.76, Recall of 0.68, and F1-score of 0.72** on the evaluated test data.

---

## Key Features

### Historical Urban Growth Analysis

Analyze changes across a temporal sequence of satellite images and extract:

- Baseline building count
- Final building count
- Newly developed buildings
- Baseline built-up area
- Final built-up area
- Monthly area change
- Cumulative urban growth
- Seasonal development trends
- Spatial development patterns

### Spatio-Temporal Prediction

The predictive framework combines three complementary deep learning components:

**CNN → Transformer → LSTM**

- CNN extracts spatial features from satellite imagery.
- Transformer models relationships between observations across time.
- LSTM captures sequential patterns in urban development.
- A prediction layer generates future building masks.

### Cloud-Aware Analysis

Optical satellite imagery can contain cloud-covered regions where buildings and land-use information become unreliable.

To address this, the framework incorporates a cloud segmentation module based on:

**U-Net + ResNet-50 encoder**

The module generates pixel-level cloud masks that identify unreliable regions before prediction.

### Multi-Step Forecasting

The system uses an auto-regressive forecasting strategy in which predicted building masks can be fed back into subsequent prediction steps, allowing the model to generate forecasts across multiple future time steps.

### Interactive Web Application

The project includes a browser-based interface that allows users to:

1. Upload a temporal stack of `.tif`/`.tiff` satellite images
2. Run the analysis pipeline
3. Explore historical urban development
4. View monthly and seasonal trends
5. View predicted building masks
6. Visualize predicted changes
7. Inspect cloud masks and overlays
8. View urban-growth and building-growth classifications

---

## System Architecture

```text
                    Multi-Temporal Satellite Images
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Data Preprocessing │
                    │                     │
                    │ • CRS verification  │
                    │ • Resizing           │
                    │ • RGB validation     │
                    │ • Normalization      │
                    │ • Spatial alignment  │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐        ┌──────────────────────┐
       │ Historical       │        │ Cloud Segmentation   │
       │ Urban Analysis   │        │                      │
       │                  │        │ U-Net + ResNet-50    │
       │ K-Means          │        │                      │
       │ Spatial Analysis │        │ → Cloud Mask         │
       │ Monthly Trends   │        └──────────┬───────────┘
       │ Seasonal Trends  │                   │
       └────────┬─────────┘                   │
                │                  ┌──────────▼───────────┐
                │                  │ CNN Spatial Features │
                │                  └──────────┬───────────┘
                │                             │
                │                  ┌──────────▼───────────┐
                │                  │ Transformer          │
                │                  │ Temporal Attention   │
                │                  └──────────┬───────────┘
                │                             │
                │                  ┌──────────▼───────────┐
                │                  │ LSTM                 │
                │                  │ Sequential Growth    │
                │                  └──────────┬───────────┘
                │                             │
                │                  ┌──────────▼───────────┐
                │                  │ Predicted Building   │
                │                  │ Mask                  │
                │                  └──────────┬───────────┘
                │                             │
                └──────────────┬──────────────┘
                               ▼
                    Urban Growth Visualization
                    and Development Classification
```

---

## Dataset

The project uses the **SpaceNet-7** multi-temporal satellite imagery dataset.

SpaceNet-7 provides high-resolution satellite imagery and building-footprint annotations for studying urban development and change over time.

### Dataset Characteristics

- Multi-temporal satellite imagery
- More than 60 urban locations
- Up to 24 monthly observations per location
- Observation period covering approximately 2018–2020
- RGB GeoTIFF satellite imagery
- Building footprints provided as GeoJSON annotations
- Ground sampling distance of approximately 30 cm

The building footprints serve as ground-truth annotations for supervised training and evaluation.

> **Note:** The SpaceNet-7 dataset is not included in this repository because of its size and dataset-specific distribution terms. Download and prepare the dataset separately before running the complete pipeline.

---

## Data Preprocessing

Before analysis and model inference, the satellite imagery undergoes several preprocessing steps.

### 1. CRS Verification

Geospatial metadata is checked using Rasterio to ensure that imagery is correctly georeferenced and spatially aligned.

### 2. Image Resizing

Satellite tiles are resized to:

```text
256 × 256 pixels
```

### 3. RGB Validation

Input images are verified to contain three RGB channels.

### 4. Spatial Consistency

Temporal images are checked to ensure that their spatial footprint remains consistent across observations.

### 5. Normalization

Pixel values are scaled to:

```text
[0, 1]
```

### 6. Building Footprint Integration

GeoJSON building annotations are rasterized and aligned with the corresponding image tiles to generate binary building masks for supervised learning and evaluation.

### Dataset Split

The research pipeline uses a non-overlapping:

| Split | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

---

# Methodology

## Phase 1 — Historical Urban Growth Analysis

The first stage focuses on understanding how urban development occurred historically.

### K-Means Spatial Clustering

Building-footprint spatial information is analyzed using K-Means clustering.

Latitude and longitude information from building polygons is used to identify spatial patterns of urban development over different time steps.

The analysis provides:

- Monthly development patterns
- Spatial density patterns
- Changes in building counts
- Changes in built-up area
- Cumulative development trends
- Seasonal construction patterns

### Monthly Analysis

For consecutive observations, the system calculates:

```text
New Buildings = Current Building Count - Previous Building Count
```

and tracks the corresponding change in built-up area.

### Seasonal Analysis

Monthly development observations are aggregated into seasonal summaries to identify variations in construction activity.

### Visualization

The analysis can be visualized through:

- Urban-density heatmaps
- Monthly development charts
- Area-change charts
- Cumulative growth trends
- Seasonal analysis
- Monthly records table

---

# Phase 2 — Hybrid Deep Learning Prediction

The predictive component uses a hybrid architecture combining spatial and temporal learning.

## CNN — Spatial Feature Extraction

The CNN processes satellite imagery and extracts spatial representations of:

- Buildings
- Roads
- Urban structures
- Local spatial patterns

The cloud information is incorporated into the predictive pipeline so that cloud-affected regions can be identified.

## Transformer — Temporal Dependency Modeling

The Transformer processes the sequence of spatial features and uses self-attention to model relationships across different observations in the temporal sequence.

This allows the model to capture long-range temporal dependencies that may not be fully represented by sequential processing alone.

## LSTM — Sequential Urban Growth Modeling

The Transformer output is passed to LSTM layers.

The LSTM models the step-by-step evolution of urban development, learning how construction activity changes across the observation period.

## Prediction Layer

The final prediction stage produces a binary building mask indicating regions where future development is expected.

---

# Phase 3 — Cloud-Aware Prediction

Cloud contamination is a major challenge when working with optical satellite imagery.

Cloud-covered regions may hide buildings and introduce unreliable information into the prediction pipeline.

### Cloud Segmentation

A U-Net-based segmentation module with a **ResNet-50 encoder** is used to identify cloud-covered regions.

```text
Satellite Image
      │
      ▼
U-Net + ResNet-50
      │
      ▼
Binary Cloud Mask
      │
      ▼
Cloud-Aware Prediction Pipeline
```

The resulting cloud mask explicitly identifies unreliable image regions.

The cloud-aware architecture therefore combines:

```text
CNN
  +
Transformer
  +
LSTM
  +
Cloud Segmentation
```

---

# Auto-Regressive Forecasting

To extend prediction beyond a single future time step, the predicted building mask can be fed back into the model as part of the next input sequence.

```text
Historical Images
       │
       ▼
Prediction t+1
       │
       ▼
Feed prediction back
       │
       ▼
Prediction t+2
       │
       ▼
Feed prediction back
       │
       ▼
Prediction t+3
       │
       ▼
       ...
```

This allows the framework to generate multi-step forecasts of future urban development.

Longer forecasting horizons can introduce accumulated prediction error, which is an important consideration when interpreting multi-step forecasts.

---

# Model Training

The research model was trained using:

- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Input:** Multi-temporal satellite imagery
- **Target:** Binary building masks

---

# Evaluation Metrics

The predictive model is evaluated against ground-truth building masks using:

### Intersection over Union (IoU)

Measures the overlap between the predicted and ground-truth building regions.

### Dice Coefficient

Measures similarity between predicted and ground-truth masks.

### Precision

Measures the proportion of predicted building pixels that correspond to actual building pixels.

### Recall

Measures the proportion of actual building pixels correctly detected.

### F1-Score

Combines precision and recall into a single metric.

---

# Results

## Proposed Model Performance

| Metric | Score |
|---|---:|
| **IoU** | **0.62** |
| **Dice** | **0.74** |
| **Precision** | **0.76** |
| **Recall** | **0.68** |
| **F1-Score** | **0.72** |

---

## Comparison with Representative Models

| Model | IoU | Dice | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| FCN | 0.42 | 0.55 | 0.60 | 0.50 | 0.54 |
| SegNet | 0.45 | 0.58 | 0.63 | 0.52 | 0.57 |
| U-Net | 0.50 | 0.65 | 0.69 | 0.58 | 0.63 |
| ConvLSTM | 0.53 | 0.68 | 0.71 | 0.60 | 0.65 |
| Transformer | 0.54 | 0.69 | 0.72 | 0.61 | 0.66 |
| **Proposed** | **0.62** | **0.74** | **0.76** | **0.68** | **0.72** |

The proposed model achieved the highest IoU, Dice, Precision, Recall, and F1-score among the compared approaches.

---

# Ablation Study

The contribution of individual architectural components was evaluated through an ablation study.

| Model Variant | IoU | Dice | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| CNN Only | 0.44 | 0.57 | 0.61 | 0.49 | 0.54 |
| CNN + LSTM | 0.49 | 0.63 | 0.67 | 0.55 | 0.60 |
| CNN + Transformer | 0.52 | 0.66 | 0.70 | 0.58 | 0.63 |
| Hybrid CNN + Transformer + LSTM | 0.56 | 0.71 | 0.73 | 0.65 | 0.69 |
| **Hybrid + Cloud Awareness** | **0.62** | **0.74** | **0.76** | **0.68** | **0.72** |

### Key observation

The progression demonstrates the contribution of the individual components:

```text
CNN
 ↓
CNN + LSTM
 ↓
CNN + Transformer
 ↓
CNN + Transformer + LSTM
 ↓
Cloud-Aware Hybrid
```

The IoU increased from **0.44** for the CNN-only model to **0.62** for the final cloud-aware hybrid framework.

The cloud-aware extension further improved IoU from **0.56 to 0.62**, indicating the value of explicitly identifying cloud-contaminated regions.

---

# Web Application

The project includes a web-based interface for interacting with the analysis pipeline.

## Workflow

```text
Upload temporal satellite-image stack
                │
                ▼
          Run Analysis
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
   Phase 1            Phase 2
Historical          Prediction
Analysis               │
        │               ▼
        │         Predicted Mask
        │
        └────────┬──────────┐
                 │          │
                 ▼          ▼
             Phase 3    Cloud Analysis
                 │
                 ▼
       Urban Growth Classification
```

## Phase 1 Interface

The frontend displays:

- Baseline area
- Final area
- Total new buildings
- Total area change
- Baseline building count
- Final building count
- Monthly building development
- Monthly area change
- Cumulative urban change
- Seasonal analysis
- Monthly records table

## Phase 2 Interface

The prediction interface displays:

- Last image in the input stack
- Predicted building mask
- Highlighted predicted changes

## Phase 3 Interface

The classification and cloud-analysis interface displays:

- Urban development classification
- Building growth classification
- Observed development
- Predicted development
- Growth percentage
- Predicted area
- Predicted buildings
- Cloud mask
- Cloud overlay

---

# Technology Stack

## Machine Learning & Data Science

- Python
- TensorFlow / Keras
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Pillow
- Rasterio
- OpenCV

## Deep Learning

- CNN
- Transformer
- LSTM
- U-Net
- ResNet-50

## Backend

- FastAPI
- Uvicorn
- Python Multipart

## Frontend

- HTML5
- CSS3
- JavaScript
- Chart.js

---

# Project Structure

```text
Urban-Development-Analysis-and-Prediction/
│
├── backend/
│   ├── main.py
│   ├── analysis.py
│   ├── model_utils.py
│   │
│   └── models/
│       └── model.h5
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Backend Components

**`main.py`**

FastAPI application responsible for:

- Receiving uploaded satellite images
- Running the analysis pipeline
- Loading the model
- Running prediction
- Running cloud segmentation
- Generating classification results
- Returning visualization data

**`analysis.py`**

Contains the historical urban-growth analysis functionality, including:

- Building-count estimation
- Built-up area estimation
- Monthly development analysis
- Seasonal aggregation

**`model_utils.py`**

Contains:

- Model loading
- Image preprocessing
- Prediction
- Cloud segmentation
- Visualization generation
- Urban-growth classification

### Frontend Components

**`index.html`**

Defines the user interface and result sections.

**`app.js`**

Handles:

- Satellite-image upload
- Temporal ordering
- API communication
- Result rendering
- Chart generation
- Prediction visualization

**`style.css`**

Defines the application's user interface and visualization styling.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/PavanKumarBaduru/Urban-Development-Analysis-and-Prediction.git
cd Urban-Development-Analysis-and-Prediction
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

Download and prepare the SpaceNet-7 dataset separately.

The application expects a temporal stack of RGB satellite images in:

```text
.tif
```

or

```text
.tiff
```

format.

### Important

The application sorts uploaded files alphabetically to preserve temporal ordering.

Therefore, filenames should follow a chronological naming convention, for example:

```text
2019_01.tif
2019_02.tif
2019_03.tif
...
2020_01.tif
```

Incorrect filename ordering can result in an incorrect temporal sequence.

---

# Running the Application

## Start the FastAPI Backend

From the `backend` directory:

```bash
cd backend
uvicorn main:app --reload
```

The API will run locally on:

```text
http://127.0.0.1:8000
```

## API Health Check

The backend provides a health endpoint:

```text
GET /health
```

A successful response is:

```json
{
  "status": "ok"
}
```

## Run Analysis

The main analysis endpoint is:

```text
POST /analyze
```

It accepts multiple satellite-image files.

At least two `.tif`/`.tiff` images are required for the temporal analysis pipeline.

---

# Frontend

After starting the backend, open:

```text
frontend/index.html
```

in a browser.

The frontend communicates with the FastAPI backend at:

```text
http://127.0.0.1:8000
```

Select multiple temporally ordered satellite images and click:

**Run Analysis**

The resulting Phase 1, Phase 2, and Phase 3 outputs are then displayed through the web interface.

---

# Current Model Artifact

The repository currently contains a placeholder `model.h5` file.

The trained model artifact used for final inference should be placed at:

```text
backend/models/model.h5
```

The backend loads the model using TensorFlow/Keras before performing prediction.

> **Note:** The trained model checkpoint is not currently included as a valid model artifact in this repository release. Replace the placeholder with the trained model before running the complete inference pipeline.

---

# Limitations

The project has several limitations identified during evaluation:

- Optical satellite imagery can still be affected by persistent cloud cover.
- Long forecasting horizons can accumulate prediction errors through auto-regressive feedback.
- The hybrid architecture has higher computational requirements than simpler segmentation models.
- Historical clustering provides a useful representation of spatial development patterns but does not capture every fine-grained structural change.
- Performance can degrade when usable imagery is sparse for extended periods.
- The current framework primarily uses RGB optical imagery.

---

# Future Work

Potential extensions include:

- Longer forecasting horizons
- Multi-spectral satellite imagery
- Synthetic Aperture Radar (SAR) data for improved cloud robustness
- More advanced cloud-aware feature fusion
- Incorporation of socio-economic factors
- Improved high-resolution urban-growth modelling
- More robust multi-step forecasting
- Deployment of the complete inference system as a scalable cloud service

---

# Key Takeaways

The project demonstrates an end-to-end approach to urban development monitoring by combining **historical spatial analysis, deep spatio-temporal modelling, cloud segmentation, and future-growth forecasting**.

The major contributions are:

1. **Historical urban-growth analysis** using clustering and spatial information.
2. **Hybrid CNN–Transformer–LSTM modelling** for spatio-temporal urban expansion prediction.
3. **Cloud-aware prediction** using U-Net and a ResNet-50 encoder.
4. **Auto-regressive forecasting** for multi-step future development prediction.
5. **Interactive FastAPI-based application** for analyzing satellite-image sequences and visualizing predictions.
6. **Comparative and ablation analysis** demonstrating the contribution of the proposed architecture.

The final cloud-aware model achieved:

> **IoU: 0.62 | Dice: 0.74 | Precision: 0.76 | Recall: 0.68 | F1: 0.72**

---

# Documentation

Detailed project documentation will be added here:

- [Project Report](docs/project-report.pdf)
- [Project Presentation](docs/project-presentation.pdf)

---

# License

A project license has not yet been specified.

---

## Project Status

**Research Prototype / Academic Project**

The repository contains the implementation of the urban development analysis and prediction application. The trained model artifact and large SpaceNet-7 dataset are maintained separately from the source repository.
