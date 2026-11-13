# AI-Powered Real-time Image Processing Platform for SPR Biosensing Applications
This repository contains all code developed for the **Development of an AI-Powered Real-time Image Processing Platform for SPR Biosensing Applications** Project. It is divided into the following components:

- `data`: This folder contains all data used in this project, divided into `Labelled` containing labelled datasets from Label-studio and `Raw` containing the raw images.
- `models`: This folder contains the training configurations and logs for the different finetuning runs (`SAM2-IoU`, `SAM2-Class`, ...). Model weights are too large for this repository and are hosted separately (see [Model weights](#model-weights)).
- `notebooks`: This folder contains various notebooks demonstrating the use of different components of the platform, divided into the main components: `biosensing_analysis`, `camera_calibration`, `finetuning`, and `segmentation`
- `src`: This folder contains the source code in plain Python files divided into three main components: `segmentation`, `finetuning`, and `analysis`.



# Dependency Setup
1. Install uv e.g. `pip install uv`
1.a if "uv is not recognized ..." error: ensure correct <python-installation>/Scripts path is added to PATH. Use `pip show uv` to show location, then add that path without the last folder but with "Scripts" to PATH.

2. Execute 
`uv sync`
to create a virtual environment and install dependencies, then 
`pip install -e .`
to enable running the notebooks and using the source code.

3. Clone and initialize SAM2 Repository (instructions from the official SAM2 repo). See "https://github.com/facebookresearch/sam2.git" for more details.
`git clone https://github.com/facebookresearch/sam2.git`
`cd sam2`
`pip install -e .`

4. Optional: download SAM2 checkpoints:
```cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

5. Optional: initialize training scripts
```
cd training && \
pip install -e ".[dev]"
```

6. Install the remaining dependencies used by `src/` and the notebooks:
`pip install torch torchvision opencv-python scipy scikit-learn scikit-image pandas pillow tqdm pycocotools imreg_dft cellpose`

7. Download the finetuned model weights (see [Model weights](#model-weights)).



# Model weights
The finetuned SAM2 checkpoint used for the results in the manuscript (`SAM2-IoU`, epoch 230, SAM2.1 Hiera Base+, ~0.9 GB) is attached to the [v1.0 release](https://github.com/benromdh/spri-automated-analysis/releases/tag/v1.0) of this repository. Download it into the location expected by the notebooks:

```bash
mkdir -p models/SAM2-IoU/checkpoints
curl -L -o models/SAM2-IoU/checkpoints/checkpoint_230.pt \
  https://github.com/benromdh/spri-automated-analysis/releases/download/v1.0/checkpoint_230.pt
```

SHA-256: `e6fe4bc3adb749265ea4c466c4392dec65d1133b5a802eca49040fa945cbe8d6`

The pretrained SAM2.1 base checkpoints (`sam2/checkpoints/sam2.1_hiera_*.pt`) are obtained with step 4 above.



# Data Directory Structure (`data/`)
## Raw
This directory contains the main datasets used throughout the project:  
- `Water-Nacl`: This is the first dataset used to find optimal camera settings, containing ~900 images across varous settings and chip regions and from 0-19% NaCl
- `Chip2-NaCl 0-19_`: This is the main dataset used for analysis and results, as it contains the best batch of images so far. Contains 81 images.
- `Chip4-NaCl 0-19_`: This dataset contains 27 images from chip 4 with good optical settings but poor experimental conditions, resulting in bad analysis results. It was used for finetuning/segmentation tests but not really for the sensitivity analysis.
- `Chip-schematic`: This folder contains the chip mask split into the 3 chip regions and whole. The first number in the filenames denotes the respective chip region, while the second number denotes the scaling parameter used when exporting from the chip design software.

## Labelled 
This directory contains the labelled datasets used for finetuning and model testing:
- `chip2_81`: This is the labelled dataset corresponding to the 81 image chip 2 dataset described above.
- `chip2_81_preprocessed`: This is the labelled chip 2 dataset with perspective correction applied to the images and annotations.
- `chip4_27`: This is the labelled dataset corresponding to the 27 image chip 4 dataset described above.
- `chip4_27_preprocessed`: This is the labelled chip 4 dataset with perspective correction applied to the images and annotations.
- `mixed_45`: This contains a selection of 45 labelled images from the `Water-Nacl`dataset described above.
- `finetuning`: This contains the training/validation/testing split that was used for the finetuning runs presented in the thesis, constructued from the chip2 and chip4 datasets (including preprocessed datasets).

## Adding a New Dataset
To add a new raw dataset, simply upload the folder containing the images to the `Raw` directory.

To add a **labelled** dataset, first create a new directory in `data/Labelled/` and import the images into a subdirectory called `Raw`and the exported .json annotations into the just created directory. See `notebooks/finetuning` section for details on how to create the SA1B structure used for finetuning.





# Models (`models/`)
This folder is used to store model information from finetuning runs. All directories were created automatically by specifying the respective path in the finetuning configuration (they are only created when the training run is started. See `notebooks/finetuning` for more details. They contain the following structure:
- `checkpoints`: This folder stores the checkpoints (model weights) created during training as .pt files. The most recent checkpoint is simply called `checkpoint.pt`, while checkpoint created during training at a certain epoch are named `checkpoint_<epoch>.pt`, e.g. `checkpoint_30.pt`
- `logs`: This directory contains logging information, specifically a `log.txt` file that is very useful for tracking the progress of the training. It contains epoch information, loss information, error messages, etc.
- `tensorboard`: This directory contains tensorboard files that can be used for visualizing the training loss and other parameters during and after training, directly in tensorboard.
- `config_resolved.yaml`: This is a copy of the finetuning configuration used for the training run.
- `config.yaml`: This is also a copy of the finetuning configuration used for the training run.




# Notebooks
## Sensitivity Analysis Notebook - Complete Processing Pipeline (`biosensing_analysis/`)
### Overview
This comprehensive notebook implements the entire image processing and analysis pipeline for refractive index sensitivity measurements. It combines preprocessing, segmentation, postprocessing, and sensitivity analysis into a unified workflow controlled through a detailed configuration system.

### Pipeline Components

### 1. Experiment Setup
The entire pipeline is controlled through `EXPERIMENT_CONFIG`, a comprehensive configuration dictionary that includes:

**Data Configuration:**
- `data_dir`: Image directory path
- `output_path`: CSV file for tracking results across experiments
- `image_batch`: Specific batch identifier (e.g., "290-4-0.75x-1")

**Camera Settings:**
- `zoom`: Camera zoom factor (0.75)
- Microns per pixel: 6.45 (hardcoded)

**Preprocessing Options:**
- `cc`: Enable camera calibration (perspective correction)
- `ffc`: Enable flatfield correction

**Model Configuration:**
- `model`: Choose between "SAM2" or "CellposeSAM"
- `sam2_config`: Path to SAM2 configuration file
- `sam2_checkpoint`: Path to model checkpoint (pretrained or finetuned)

**Postprocessing Settings:**
- `approx`: Use approximated masks (circle/ellipse fitting)
- `approx_method`: "circle" or "ellipse" approximation
- `shrinking_factor`: Different scaling factors for large, medium, small wells

**Analysis Options:**
- `overlay_masks`: Use masks from first image on all subsequent images
- `individual_intensities`: Show individual intensity points in plots

### 2. Data Loading and Object Initialization

**Data Processing:**
- Loads images from specified directory using batch filtering
- Groups images by increasing NaCl concentration
- Initializes camera object with zoom and pixel size parameters

**Model Setup:**
- Configures SAM2 or CellposeSAM wrapper
- Creates Segmenter object with detailed filter configuration
- Sets up 128 points per side for SAM2 automatic mask generation

### 3. Preprocessing Pipeline

**Camera Calibration Setup:**
- Loads chip schematics for three different regions
- Extracts binary masks from schematics
- Initializes CameraCalibrator with reference radii for each region
- Supports both individual and batch transformation calculation

**Image Enhancement:**
- Applies flatfield correction to reduce illumination artifacts
- Performs perspective correction using affine and homography transformations
- Processes all images in batch with progress tracking
- Stores both original and enhanced images

### 4. Image Segmentation

**Segmentation Process:**
- Generates masks using chosen model (SAM2 or CellposeSAM)
- Applies comprehensive filtering based on:
 - Circularity threshold (0.7)
 - Border distance requirements
 - Area-based clustering (DBSCAN)
 - Coordinate-based clustering for different well sizes
- Separates masks into large, medium, and small well categories

**Mask Overlay Option:**
- Can use masks from first image across all images in series
- Useful for temporal analysis where well positions remain constant

### 5. Mask Postprocessing

**Geometric Approximation:**
- Applies circle or ellipse fitting to segmented masks
- Uses different shrinking factors for each well size:
 - Large wells: 0.9
 - Medium wells: 0.8
 - Small wells: 0.6
- Optimized for Chip2 data characteristics

### 6. Intensity Analysis

**Measurement Calculation:**
- Extracts mean intensity values from each well
- Calculates standard deviation for error estimation
- Processes all well size categories independently
- Uses enhanced (preprocessed) images for measurements

### 7. Sensitivity Analysis and Results

**Data Compilation:**
- Creates comprehensive DataFrame with:
 - NaCl concentrations
 - Mean and standard deviation for each well size
 - Combined metrics across all well sizes
 - Refractive index calculation: RI = 1.3330 + 0.0018 × NaCl%

**Visualization and Analysis:**
- **Mask Visualization**: Shows segmentation results with optional well identification
- **Intensity vs RI Plots**: Linear regression analysis for each well size
- **Sensitivity Metrics**: Calculates slope of intensity vs refractive index relationship
- **Statistical Analysis**: Computes average standard deviations across well sizes

**Results Storage:**
- Saves detailed results to CSV file for cross-experiment comparison
- Tracks different preprocessing combinations (ffc, cc, approx)
- Stores complete plot recreation data including error bars and NaCl percentages

## Key Features

**Comprehensive Configuration:**
- Single configuration dictionary controls entire pipeline
- Easy parameter tuning for optimization studies
- Supports multiple model types and preprocessing combinations

**Robust Processing:**
- Handles multiple chip regions with different characteristics
- Includes extensive debugging and visualization options
- Progress tracking for long processing runs

**Advanced Analysis:**
- Linear regression analysis with R² values
- Individual well tracking capabilities
- Cross-well-size sensitivity comparison
- Automated results compilation and storage

**Flexibility:**
- Supports both batch and individual image processing
- Configurable mask overlay for temporal studies
- Multiple approximation methods for different use cases

## Output
- Comprehensive sensitivity analysis results
- Linear regression slopes for each well size category
- Visual confirmation of segmentation quality
- Detailed CSV logs for experiment tracking and comparison





## Segmentation and Model Testing (`segmentation/`)
### Overview
This Jupyter notebook provides a framework for evaluating segmentation models using a standardized evaluation pipeline. It's designed to compare different model configurations and generate performance metrics.

### Purpose
- **Model Evaluation**: Test and compare different segmentation models
- **Performance Metrics**: Generate quantitative results for model comparison
- **Experiment Tracking**: Organize and store evaluation results systematically

### Outputs
- Saves evaluation results to: `output_path/experiment_name_summary.csv`
- Contains performance metrics and comparison data

### Usage Instructions

**1. Configuration Setup**
For example ```
experiment_name = "model_test"  # Set experiment identifier
output_path = "./"              # Define results storage location
dataset = "test/combined"       # Choose dataset ("test/combined" or "val")
```
**2. Dataset Paths**
For example ```
im_dir = f"../../data/Labelled/finetuning/{dataset}/Images"
ann_dir = f"../../data/Labelled/finetuning/{dataset}/Annotations"
```
**3. Model Selection**
Load the correct model and change the `model` parameter of the `segmenter` object. Multiple models used throughout the thesis have already been loaded.




## Perspective Distortion Evaluation (`camera_calibration/`)
### Overview
This notebook evaluates the effectiveness of camera calibration (perspective correction) by comparing segmentation performance before and after applying geometric corrections. It quantifies improvement using Intersection over Union (IoU) metrics across different well sizes.

### Purpose
- **Calibration Assessment**: Measure how well camera calibration improves image geometry
- **Segmentation Validation**: Compare segmentation accuracy on corrected vs uncorrected images
- **Quantitative Metrics**: Use IoU scores to objectively evaluate calibration effectiveness
- **Cross-Well Analysis**: Assess calibration impact on different well sizes independently

### Experimental Setup

### Configuration Parameters
The experiment uses a comprehensive configuration system similar to the sensitivity analysis:

**Data Configuration:**
- `data_dir`: Raw image directory path
- `image_batch`: Specific chip region identifier (e.g., "290-4-0.75x-3")
- `zoom`: Camera zoom factor (0.75)

**Processing Options:**
- `cc`: Camera calibration enabled (True for evaluation)
- `ffc`: Flatfield correction (disabled for isolated testing)
- `clahe`: Contrast Limited Adaptive Histogram Equalization (disabled)

**Model Settings:**
- `model`: Uses "CellposeSAM" for segmentation
- `approx`: Enables geometric approximation (ellipse fitting)
- `shrinking_factor`: Conservative 0.4 factor for all well sizes

**Calibration Configuration:**
- Region-specific reference radii for alignment
- Schematic paths for three different chip regions
- Batch processing mode (not individual image calculation)

### Methodology

### 1. Object Initialization
**Camera Setup:**
- Configures camera with zoom and pixel size parameters
- Initializes with 6.45 microns per pixel resolution

**Segmentation Model:**
- Uses CellposeSAM wrapper for consistent segmentation
- Applies filtering based on circularity, border distance, and clustering
- Separates wells into large, medium, and small categories

**Calibrator Setup:**
- Loads chip schematics for reference alignment
- Extracts binary masks from schematic images
- Configures region-specific parameters and reference radii
- Sets up transformation matrices for perspective correction

### 2. Evaluation Process
**For Each Image:**
1. **Load and Process**: Opens grayscale image from dataset
2. **Apply Corrections**: Runs both affine and full perspective correction
3. **Visual Inspection**: Displays corrected image for quality assessment
4. **Dual Evaluation**: Tests segmentation on both corrected versions
5. **IoU Calculation**: Compares segmentation results against reference schematic

**Transformation Steps:**
- **Affine Correction**: Basic geometric transformation (rotation, scaling, translation)
- **Perspective Correction**: Full homography correction for lens distortion

### 3. Metrics Collection
**IoU Calculations:**
- `iou_all`: Overall intersection over union across all detected wells
- `iou_large`: IoU specifically for large wells
- `iou_medium`: IoU for medium-sized wells  
- `iou_small`: IoU for small wells

**Data Structure:**
- DataFrame stores results for both correction levels per image
- Even rows: Affine correction results
- Odd rows: Full perspective correction results

### 4. Results Analysis
**Statistical Processing:**
- Calculates averages for even-indexed rows (affine correction)
- Calculates averages for odd-indexed rows (perspective correction)
- Provides direct comparison between correction methods

**Output Generation:**
- Saves complete results to CSV file
- Includes individual image results and summary statistics
- Provides quantitative assessment of calibration effectiveness

### Key Features

**Controlled Comparison:**
- Tests identical images with and without calibration
- Uses consistent segmentation parameters across tests
- Isolates camera calibration effects from other processing steps

**Multi-Level Assessment:**
- Evaluates both affine and perspective correction stages
- Provides well-size-specific performance metrics
- Offers comprehensive view of calibration impact

**Quantitative Validation:**
- Uses IoU as objective measure of geometric accuracy
- Compares against known reference (chip schematic)
- Provides statistical summary of improvement

**Visual Confirmation:**
- Displays corrected images for qualitative assessment
- Enables visual validation of geometric improvements
- Supports debugging of calibration parameters

## Expected Outcomes
- **Improved IoU Scores**: Higher values after calibration indicate better geometric accuracy
- **Well-Size Variations**: Different improvement levels across large, medium, and small wells
- **Method Comparison**: Quantifies difference between affine and perspective correction
- **Calibration Validation**: Confirms effectiveness of camera calibration approach

### Applications
- **Parameter Tuning**: Optimize calibration settings for specific chip regions
- **Method Validation**: Verify calibration algorithm effectiveness
- **Quality Control**: Ensure consistent geometric correction across experiments
- **Comparative Analysis**: Evaluate different correction approaches quantitatively




## Training and Validation Notebooks (`finetuning/`)
### 1. train_like_SA1B.ipynb - SAM2 Finetuning Guide

### Overview
A simple notebook that demonstrates how to finetune SAM2 using Meta's official training framework with SA-1B dataset format.

### Prerequisites
- Run from top-level SAM2 directory
- Install SAM2 and SAM2.dev: `pip install -e ".[dev]"` and `pip install -e .`
- Config file stored at `configs/sam2.1_training/...`

### Usage
Execute the training command with customizable parameters:
- **Config**: Path to YAML configuration file
- **Cluster**: Set to 0 for local training
- **GPUs**: Number of GPUs to use

### Features
- Uses Meta's official SAM2 training framework
- TensorBoard integration for monitoring training progress
- Configurable through YAML files

### 2. train_test_split.ipynb - Dataset Split Creation

### Overview
Creates train/validation/test splits specifically designed for chip2 and chip4 datasets used in thesis work.

### Dataset Sources
- **chip2_path**: Original chip2 dataset (81 samples)
- **chip4_path**: Original chip4 dataset (27 samples)
- **Preprocessed versions**: Enhanced versions of both datasets

### Configuration
Uses configurable ratios: 70% training, 15% validation, 15% testing

### Features
- Balanced splitting across different chip types
- Handles both original and preprocessed datasets
- Customizable split ratios
- Organized output structure

### 3. turn_into_SA1B_dataset.ipynb - SA-1B Dataset Conversion

### Overview
Converts Label Studio ellipse annotations (.json) and images into SA-1B format compatible with SAM2's training framework.

### Input Requirements
- **Images**: Raw image folder
- **Annotations**: Label Studio JSON annotations with ellipse format

### Process Flow
1. **Setup Paths**: Configure input/output directories
2. **Convert Annotations**: Transform Label Studio format to SA-1B format
3. **Create Dataset**: Generate SA-1B compatible structure
4. **Visualization**: Verify conversion with mask overlays

### Key Functions
- `construct_masks_from_json()`: Parse Label Studio annotations
- `create_sa1b_dataset()`: Generate SA-1B format structure
- `visualize_annotation()`: Quality check converted masks

### Features
- Handles filename compatibility issues
- Maintains annotation quality through visualization
- Compatible with Meta's SA-1B dataloader




## 4. validation.ipynb - Model Validation Framework

### Overview
Comprehensive validation system for evaluating multiple SAM2 model checkpoints across different datasets.

### Model Configurations
- **SAM2-Mask**: Mask-based loss training
- **SAM2-Class**: Classification-focused training  
- **SAM2-IoU**: IoU-optimized training
- **SAM2-Dice**: Dice coefficient-based training

### Dataset Variants
- **Combined**: Mixed chip2 and chip4 data
- **Chip-specific**: Individual chip2 or chip4 datasets
- **Preprocessed**: Enhanced versions of datasets

### Validation Process
1. **Multi-model Testing**: Evaluate all model configurations
2. **Cross-dataset Validation**: Test on multiple dataset variants
3. **Checkpoint Analysis**: Compare performance across training epochs
4. **Results Visualization**: Generate training curves and performance plots

### Outputs
- CSV files with detailed metrics for each checkpoint
- Training curve visualizations
- Comparative performance analysis
- Best checkpoint identification

### Features
- Automated batch validation across multiple models
- Comprehensive performance tracking
- Visual analysis of training progression
- Structured results storage for further analysis


# Source Code Structure (`src/`)
## Segmentation Module
**eval_utils.py**  
Core evaluation utilities for comparing predicted masks against ground truth. Contains functions for filtering overlapping masks, converting annotations to well masks, computing IoU metrics, and calculating precision/recall/F1 scores.

**filtering.py**  
Mask filtering functions using DBSCAN clustering. Includes cluster_area() for grouping masks by area and cluster_coords() for spatial clustering to remove noise and outliers from segmentation results.

**mask_container.py**  
Container class for managing collections of well masks. Provides filtering methods (circularity, area, duplicates, border), visualization capabilities, and well size approximation using K-means clustering.

**model_evaluation.py**  
Main evaluation pipeline for segmentation models. Compares SAM2 predictions against ground truth annotations, calculates comprehensive metrics (precision, recall, F1, IoU) by well size, and generates CSV reports.

**segmenter.py**  
Main wrapper classes for segmentation models including SAM2Wrapper, CellposeSAMWrapper, and hybrid approaches. Handles mask generation, filtering pipelines, and visualization of results by well size categories.

**utils.py**  
Helper functions for mask processing including circularity computation, K-means clustering for well grouping by area, sigma filtering for outlier removal, and mask list manipulation utilities.

**well_mask.py**  
Core WellMask class representing individual segmented wells. Includes properties for area, bounding box, center calculation, contour extraction, and mask approximation methods (circle/ellipse) with visualization capabilities.

**finetuning_utils.py**  
Utilities for processing Label Studio annotations. Converts ellipse annotations from JSON format to RLE masks, handles coordinate transformations, and includes helper functions for image name preprocessing.

## Finetuning Module
**sa1b_dataset_utils.py**  
Utilities for processing and converting Label Studio annotations into SAM2 (Segment Anything Model) compatible format (specifically into the SA1B format). This module handles:
- Converting ellipse annotations from Label Studio JSON exports to COCO-format RLE masks
- Creating SA-1B (Segment Anything 1 Billion) formatted datasets with sequential naming
- Drawing ellipse masks and computing segmentation properties (area, bounding boxes)
- Visualizing segmentation annotations overlaid on original images
- Handling filename normalization issues from Label Studio exports

Key functions include `construct_masks_from_json()` for annotation processing, `create_sa1b_dataset()` for dataset organization, and `visualize_annotation()` for quality control visualization.

**train_test_split.py**  
Creates stratified train/validation/test splits for SAM2 model training while preventing data leakage. This module:
- Splits multiple related datasets (chip2, chip4, and their preprocessed versions) consistently
- Ensures corresponding files exist across original and preprocessed datasets
- Creates both individual dataset splits and combined datasets for unified training
- Generates file lists compatible with SAM2 training scripts
- Maintains proper directory structure with prefixed filenames to avoid conflicts

The main function `create_dataset_split()` produces organized train/val/test directories with comprehensive metadata for reproducible model training.

**validation.py**  
Comprehensive checkpoint evaluation and training progress analysis for SAM2 models. This module provides:
- Systematic evaluation of all model checkpoints to track training progress
- Loading and analysis of validation results across multiple epochs
- Visualization of training curves (precision, recall, F1, IoU) for different object sizes
- Identification of best-performing checkpoints for deployment
- Robust error handling for failed evaluations

Key functions include `validate_checkpoints()` for automated checkpoint testing, `analyze_checkpoint_results()` for complete analysis pipeline, and plotting utilities for publication-ready training curve visualizations.

## Analysis Module
**analysis.py**  
Contains functions for intensity analysis of segmented image regions. The main function `intensity_analysis()` calculates mean and standard deviation of pixel intensities within masked areas. Includes a helper function `calculate_intensity()` that extracts intensities from grayscale images using mask containers, and applies a two-sigma filter to remove outliers from the intensity measurements.

**camera_calibration.py**  
Implements camera calibration and image distortion correction for microscopy applications. The main `CameraCalibrator` class handles affine transformations and homography calculations to correct optical distortions. It includes methods for aligning reference images with actual camera captures, calculating transformation matrices, and applying geometric corrections. Also contains utility functions for image registration, mask matching, and evaluation of calibration quality using IoU metrics.

**data_preprocessing.py**  
Provides utilities for organizing and preprocessing image datasets. The main function `data_preprocessing()` scans a directory of images and extracts experimental parameters (like NaCl concentration) from filenames, organizing them into structured dictionaries grouped by experimental conditions. Includes filename parsing functions that extract concentration values and batch identifiers from standardized naming conventions.

**noise_correction.py**  
Implements flat-field correction algorithms to remove illumination non-uniformities and noise from microscopy images. The `FlatFieldCorrector` class provides methods to estimate dark-field and flat-field images, then apply corrections to normalize illumination across the image. Includes visualization functions to compare original and corrected images side-by-side.

**utils.py**  
Contains various utility functions for image processing workflows including preprocessing pipelines, visualization tools, and data analysis helpers. Key functions include `apply_preprocessing()` for combining different correction methods, `create_mask_plots()` and `create_intensity_plots()` for visualization, and `plot_intensity_vs_nacl()` for analyzing the relationship between refractive index and image intensity. Also includes CSV export functionality and timing decorators for performance monitoring.

## Importing Modules 
To use code from any of the modules, simply add an import line like `from segmentation.segmenter import *`, then you can use all available classes from the **segmenter.py** file.




# Using a New Model
To use a new model in the pipeline, you first have to create a wrapper class that exposes a `generate_masks()` function for it. This function should implement the model-specific logic for generating masks based on an image and return a `MaskContainer` with the detected masks. See `src.segmentation.segmenter.SAM2Wrapper` for an example.

## 1. Create Model Wrapper Class

Create a new wrapper class that inherits from the base model interface and implements the required methods:

```python
class YourModelWrapper:
   def __init__(self, config_path=None, checkpoint_path=None, **kwargs):
       # Initialize your model with necessary parameters
        ...
   
   def generate_masks(self, image):
       # Generate segmentation masks for input image
       ...
       masks = self.model.predict(image)
       return masks
```

## 2. Integrate into Pipeline

Once the wrapper exists, you have to instantiate an object of the wrapper class and pass it to the segmenter class as used in almost every notebook (see `notebooks/`). Once passed to the segmenter class, it will integrate seamlessly with the existing pipeline.


# Strating a new Finetuning Run
## Overview
This comprehensive guide walks through the entire process of finetuning SAM2 models, from importing labeled datasets to training execution and evaluation. The workflow ensures proper data preparation, training configuration, and performance assessment.

## Prerequisites

### Environment Setup
- SAM2 repository cloned and properly installed
- Required dependencies: '''pip install -e ".[dev]"''' and '''pip install -e .'''
- GPU with sufficient memory (recommended: 96GB, this is the max for a single GPU on the cluster)
- Adequate storage space for datasets and checkpoints
- Esnrue that enough directory space is free on your HPC server, as saved chackpoints during training are quite big (~800MB each)

### Required Data
- **Raw Images**: Original microscopy images
- **Label Studio Annotations**: JSON file with ellipse annotations

## Step 1: Dataset Preparation

### 1.1 Organize Raw Data Structure
Ensure your data follows this structure:
```
data/
├── Raw/
│   ├── Images/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── annotations.json
```

### 1.2 Convert to SA-1B Format
Use the ```turn_into_SA1B_dataset.ipynb``` notebook:

**Configuration:**
```python
BASE_DIR = "../../data/Labelled/your_dataset_name"
INPUT_IMAGE_DIR = os.path.join(BASE_DIR, "Raw")
INPUT_ANNOTATION_FILE = os.path.join(BASE_DIR, "annotations.json")
OUTPUT_IMAGE_DIR = os.path.join(BASE_DIR, "LikeSA1B", "Images")
OUTPUT_ANNOTATION_DIR = os.path.join(BASE_DIR, "LikeSA1B", "Annotations")
```

**Execution Steps:**
1. Run ```construct_masks_from_json(INPUT_ANNOTATION_FILE)```
2. Execute ```create_sa1b_dataset(data, INPUT_IMAGE_DIR, OUTPUT_IMAGE_DIR, OUTPUT_ANNOTATION_DIR)```
3. Verify conversion with ```visualize_annotation(elem, OUTPUT_IMAGE_DIR)```

**Expected Output:**
```
your_dataset_name/
├── LikeSA1B/
│   ├── Images/
│   │   ├── image001.jpg
│   │   └── ...
│   └── Annotations/
│       ├── image001.json
│       └── ...
```

### 1.3 Create Train/Validation/Test Split
Use '''train_test_split.ipynb''' for systematic data splitting (this function is specific to the chip2 and chip4 datasets. Alternatively write a custom function to generate a new train/val/test split:

**Configuration Example:**
```python
chip2_path = "../../data/Labelled/chip2_dataset"
chip4_path = "../../data/Labelled/chip4_dataset"
chip2_preprocessed_path = "../../data/Labelled/chip2_preprocessed"
chip4_preprocessed_path = "../../data/Labelled/chip4_preprocessed"
output_path = "../../data/Labelled/finetuning"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
```

**Execution:**
```python
create_dataset_split(
   chip2_path=chip2_path,
   chip4_path=chip4_path,
   chip2_preprocessed_path=chip2_preprocessed_path,
   chip4_preprocessed_path=chip4_preprocessed_path,
   output_path=output_path,
   train_ratio=train_ratio,
   val_ratio=val_ratio,
   test_ratio=test_ratio
)
```

## Step 2: Training Configuration

### 2.1 Basic Training Parameters
Create a YAML configuration file at '''configs/sam2.1_training/your_experiment.yaml''':

```yaml
# @package _global_

scratch:
 resolution: 1024              # Input image resolution
 train_batch_size: 1           # Batch size (adjust based on GPU memory)
 num_train_workers: 1          # Number of data loading workers
 num_frames: 1                 # Number of frames per sample (1 for static images)
 max_num_objects: 150          # Maximum objects per frame
 base_lr: 5.0e-06             # Base learning rate
 vision_lr: 3.0e-06           # Vision encoder learning rate (usually lower)
 phases_per_epoch: 1          # Training phases per epoch
 num_epochs: 500              # Total training epochs
```

### 2.2 Dataset Configuration
Update paths to point to your prepared dataset:

```yaml
dataset:
 # Update these paths to your dataset location
 img_folder: /path/to/your/data/train/combined/Images
 gt_folder: /path/to/your/data/train/combined/Annotations
 file_list_txt: /path/to/your/data/train/train_combined_files.txt
 multiplier: 10               # Dataset multiplier for epoch length
```

### 2.3 Data Augmentation Settings
Configure training transforms for data augmentation:

```yaml
vos:
 train_transforms:
   - _target_: training.dataset.transforms.ComposeAPI
     transforms:
       - _target_: training.dataset.transforms.RandomHorizontalFlip
         consistent_transform: false
       - _target_: training.dataset.transforms.RandomAffine
         degrees: 15            # Rotation range
         shear: 10             # Shear transformation
         image_interpolation: bilinear
         consistent_transform: false
       - _target_: training.dataset.transforms.RandomResizeAPI
         sizes: ${scratch.resolution}
         square: true
         consistent_transform: false
       - _target_: training.dataset.transforms.ColorJitter
         consistent_transform: false
         brightness: 0.15      # Brightness variation
         contrast: 0.1         # Contrast variation
         saturation: 0.0       # Saturation (0.0 for grayscale images)
         hue: 0.0             # Hue variation (0.0 for grayscale)
       - _target_: training.dataset.transforms.ToTensorAPI
       - _target_: training.dataset.transforms.NormalizeAPI
         mean: [0.449, 0.449, 0.449]  # Normalization mean
         std: [0.226, 0.226, 0.226]   # Normalization std
```

### 2.4 Loss Function Configuration
Configure loss functions based on your training objective:

**Multi-Loss Configuration (Recommended):**
```yaml
trainer:
 loss:
   all:
     _target_: training.loss_fns.MultiStepMultiMasksAndIous
     weight_dict:
       loss_mask: 1           # Binary mask loss weight
       loss_dice: 10          # Dice loss weight
       loss_iou: 1            # IoU loss weight
       loss_class: 1          # Classification loss weight
     supervise_all_iou: true
     iou_use_l1_loss: true
     pred_obj_scores: true
     focal_gamma_obj_score: 2.0
     focal_alpha_obj_score: 0.25
```

### 2.5 Checkpoint and Logging Configuration
Set up model initialization and logging:

```yaml
trainer:
 logging:
   tensorboard_writer:
     _target_: training.utils.logger.make_tensorboard_logger
     log_dir: ${launcher.experiment_log_dir}/tensorboard
     flush_secs: 120
     should_log: True
   log_dir: ${launcher.experiment_log_dir}/logs
   log_freq: 10

 checkpoint:
   save_dir: ${launcher.experiment_log_dir}/checkpoints
   save_freq: 10              # Save checkpoint every 10 epochs
   model_weight_initializer:
     _partial_: True
     _target_: training.utils.checkpoint_utils.load_state_dict_into_model
     strict: True
     state_dict:
       _target_: training.utils.checkpoint_utils.load_checkpoint_and_apply_kernels
       checkpoint_path: ./checkpoints/sam2.1_hiera_base_plus.pt
       ckpt_state_dict_keys: ['model']

launcher:
 num_nodes: 1
 gpus_per_node: 1
 experiment_log_dir: /path/to/your/experiment/output/directory
```

### 2.6 Training-Specific Parameters
Configure training behavior and sampling:

```yaml
trainer:
 model:
   # Training specific parameters
   prob_to_use_pt_input_for_train: 1.0      # Probability to use point input
   prob_to_use_box_input_for_train: 0.0     # Probability to use box input
   prob_to_sample_from_gt_for_train: 0.3    # Sample correction points from GT
   num_frames_to_correct_for_train: 1       # Frames to apply corrections
   rand_frames_to_correct_for_train: True   # Random frame correction
   num_init_cond_frames_for_train: 1        # Initial conditioning frames
   num_correction_pt_per_frame: 1           # Correction points per frame
```

## Step 3: Execute Training

### 3.1 Navigate to SAM2 Directory
```bash
cd /path/to/sam2/repository
```

### 3.2 Launch Training
Use ```train_like_SA1B.ipynb``` or run directly from command line:

**From Notebook:**
```python
!python training/train.py \
   --config "configs/sam2.1_training/your_experiment.yaml" \
   --use-cluster 0 \
   --num-gpus 1
```

**From Command Line:**
```bash
python training/train.py \
   --config "configs/sam2.1_training/your_experiment.yaml" \
   --use-cluster 0 \
   --num-gpus 1
```

### 3.3 Monitor Training Progress
**TensorBoard Monitoring (needs special setup in jupyterhub, never worked for me):**
```bash
tensorboard --logdir ./outputs/your_experiment_name/tensorboard
```

**Key Metrics to Monitor:**
- Training loss convergence

## Step 4: Validation During Training

### 4.1 Configure Validation Framework
Use '''validation.ipynb''' to systematically evaluate checkpoints:

**Setup Validation Paths:**
```python
configs = ["your_experiment_name"]

im_dirs = [
   "../../data/Labelled/finetuning/val/combined/Images",
   "../../data/Labelled/finetuning/val/chip2/Images",
   "../../data/Labelled/finetuning/val/chip4/Images"
]

ann_dirs = [
   "../../data/Labelled/finetuning/val/combined/Annotations",
   "../../data/Labelled/finetuning/val/chip2/Annotations", 
   "../../data/Labelled/finetuning/val/chip4/Annotations"
]

checkpoint_dir = f"../../models/{model-name}/checkpoints"
```

### 4.2 Run Checkpoint Validation
```python
for training_config in configs:
   checkpoint_dir = f"../../models/{training_config}/checkpoints"
   
   for im_dir, ann_dir, output_dir in zip(im_dirs, ann_dirs, output_dirs):
       result = validate_checkpoints(
           image_dir=im_dir,
           ann_dir=ann_dir,
           checkpoint_dir=checkpoint_dir,
           output_dir=output_dir
       )
```

### 4.3 Analyze Validation Results
```python
directory_path = "./ValidationResults/your_experiment_name/combined"

results = analyze_checkpoint_results(
   directory_path=directory_path,
   save_plot="training_curves.png"
)
```

## Step 5: Final Model Evaluation

### 5.1 Select Best Checkpoint
Based on validation results, identify the optimal checkpoint:
- Best validation performance
- Best performance on target metrics
- Stable convergence indicators

### 5.2 Comprehensive Evaluation
Use ```segmentation_eval.ipynb``` for thorough assessment:

**Configuration:**
```python
experiment_name = "final_evaluation_your_model"
output_path = "./evaluation_results/"
dataset = "test/combined"

checkpoint_path = "../../models/your_experiment/checkpoints/checkpoint_best.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

sam2 = SAM2Wrapper(
   config=model_cfg, 
   checkpoint=checkpoint_path, 
   points_per_side=128
)

segmenter = Segmenter(model=sam2, filter_config={
   "circularity_threshold": 0.7,
   "border_thres": 2,
   "max_groups": None,
   "eps_area": 40,
   "eps_coords_medium": 90,
   "eps_coords_small": 70,
   "rough_only": False,
   "debug": True
})
```

**Run Evaluation:**
```python
model_comparer(
   image_path=im_dir,
   annotation_path=ann_dir,
   segmenter=segmenter,
   experiment_dir=output_path,
   experiment_name=experiment_name
)
```

### Troubleshooting Common Issues

**Training Crashes:**
- Reduce max_num_objects if out of memory or request more memory (if not already at 96GB per GPU)
- Check data paths and file integrity
- Verify CUDA compatibility

By following this comprehensive workflow, you can successfully finetune SAM2 models for your specific segmentation tasks while maintaining proper evaluation and validation procedures.

