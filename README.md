# Hypothesis Test-Based Particle Detection

A Python-based particle detection and counting system for nanoparticle assay images using a multiple-hypothesis testing approach and information-theoretic model selection.


**Publication:** Details of the algorithm will be available upon publication. 

NB/ Whilst current implementation contains some RGB functionality, only the grayscale functionality has been fully implemented and tested. Use RGB capabilities at your own peril. 

## Description

This project provides a comprehensive framework for detecting and counting fluorescent spots or scattering nanoparticles in microscopy images through a multiple-hypothesis testing approach. The system can:

- **Generate synthetic test datasets** with configurable particle counts, PSF properties, and noise characteristics
- **Analyze experimental or synthetic images** to detect and count particles
- **Preprocess experimental data** by dividing large images into analyzable tiles and estimating PSF parameters
- **Evaluate algorithm performance** through confusion matrices, accuracy metrics, and statistical analysis

The core algorithm uses maximum likelihood estimation under different hypotheses (0 particles, 1 particle, 2 particles, etc.) and selects the best model using an information-theoretic criterion.

## Repository Structure

### Root Directory - Core Analysis Files

```
├── main.py                      # Main entry point for dataset generation and analysis
├── process_algorithms.py        # Core GLRT algorithm implementation
├── image_generation.py          # Synthetic image generation with PSF convolution
├── preprocess_exp_data.py       # Experimental data preprocessing and PSF estimation
└── requirements.txt             # Python package dependencies
```

### Main Entry Points

- **`main.py`**: Primary workflow orchestrator
  - Reads JSON config files from `./configs/` or subdirectories
  - Generates synthetic datasets (optional)
  - Analyzes images in `./datasets/` folders
  - Outputs results to `./analyses/` with confusion matrices, logs, and metrics
  - Supports parallel processing with timeout handling

- **`preprocess_exp_data.py`**: Experimental data preparation
  - Divides large TIFF images into smaller tiles with optional overlap
  - Performs Gaussian fitting on selected tiles to estimate PSF width (sigma)
  - Generates JSON config files for subsequent analysis
  - Supports both UI and command-line modes

- **`process_algorithms.py`**: Algorithm implementation
  - `generalized_maximum_likelihood_rule()`: Main generalized maximum likelihood rule function
  - `merge_coincident_particles()`: Combines detections from overlapping tiles
  - Particle position/intensity optimization using scipy.optimize
  - Fisher information matrix computation

### Data Directories

```
configs/                         # Configuration JSON files
example_config_folder/           # Example configuration files
datasets/                        # Generated/preprocessed image datasets
analyses/                        # Analysis output (confusion matrices, logs, metrics)
data/                            # Raw experimental data (user-provided)
```

## Installation

### Requirements

- Python 3.8+
- See `requirements.txt` for complete package list

### Setup

```powershell
# Clone the repository
Use the "Code" → "Clone" button on the GitHub page to copy the HTTPS or SSH URL, then run:
git clone <paste-clone-URL-here>

cd Hypothesis-Test-Based-Particle-Detection

# Create and activate virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Example Usage

### Case 1: Analyze Experimental Data

**Workflow:**
1. Preprocess raw TIFF images
2. Generate config files
3. Run analysis

```powershell
# Step 1: Preprocess experimental data (UI mode)
python preprocess_exp_data.py

# Or use command-line mode with predefined PSF sigma
# (see doc string for full list of command line arguments)
python preprocess_exp_data.py --terminal 
    --folder "D:\data\exp_images" 
    --size 80 
    --overlap 0 
    --interval 1 
    --predefined-sigma 1.5 
    --config-subdir "exp_batch1"

# Step 2: Run analysis using generated configs
python main.py --config-files-dir "configs/exp_batch1" --parallel
```

**What happens:**
- `preprocess_exp_data.py` divides large images into 80×80 pixel tiles with 0px overlap (or user-supplied values)
- Interval 1 processes all images in datafolder (i.e., no file is skipped). Timestamping based filter is available but unlikely of use to most users.
- PSF sigma = 1.5. If a predefined sigma is not provided, the script will guide you to estimate it via Gaussian fitting.
- Configs saved to `./configs/exp_batch1/`
- Analysis outputs to `./analyses/` with confusion matrices and performance metrics

### Case 2: Generate and Test on Synthetic Data

**Workflow:**
1. Create config file for dataset generation
2. Run main.py to generate and analyze

**Example config** (`configs/my_test/synthetic_baseline.json`):
```json
{
    "image_folder_namebase": "synthetic_test",
    "code_version_date": "2025-05-02",
    "file_format": "tiff",
    
    "generate_regular_dataset?": true,
    "gen_random_seed": 12345,
    "gen_total_image_count": 100,
    "gen_minimum_particle_count": 0,
    "gen_maximum_particle_count": 5,
    "gen_psf_sigma": 2.0,
    "gen_img_width": 100,
    "gen_bg_level": 2000,
    "gen_particle_intensity_mean": 20000,
    "gen_particle_intensity_sd": 2000,
    
    "analyze_the_dataset?": true,
    "ana_random_seed": 54321,
    "ana_predefined_psf_sigma": 2.0,
    "ana_use_premature_hypothesis_choice?": false,
    "ana_maximum_hypothesis_index": 6,
    "ana_timeout_per_image": 120,
    "ana_delete_the_dataset_after_analysis?": false
}
```

Run:
```powershell
python main.py --config-files-dir "configs/my_test"
```

**Output:**
- Images: `./datasets/synthetic_test/count{n}_index{i}.tiff`
- Results: `./analyses/synthetic_test_code_ver2025-05-02/`
  - Confusion matrix (CSV + PNG)
  - Label-prediction log
  - Per-hypothesis metrics
  - Accuracy/RMSE scores

## Configuration File Format

JSON config files control all aspects of image generation and analysis. See `example_config_folder/d4-baseline.json` for a complete example.

### Required Fields (Common)

```json
{
    "image_folder_namebase": "dataset_name",  // Base name for folders
    "code_version_date": "YYYY-MM-DD"         // Version identifier
}
```

### Image Generation Fields

```json
{
    "generate_regular_dataset?": true,        // Enable generation
    "gen_random_seed": 0,
    "gen_total_image_count": 100,
    "gen_minimum_particle_count": 0,
    "gen_maximum_particle_count": 5,
    "gen_psf_sigma": 2.0,                     // PSF width (pixels)
    "gen_img_width": 100,                     // Image size (pixels)
    "gen_bg_level": 2000,                     // Background intensity
    "gen_particle_intensity_mean": 20000,
    "gen_particle_intensity_sd": 0            // 0 = fixed intensity
}
```

### Separation Test Generation (Alternative)

```json
{
    "separation_test_image_generation?": true,
    "sep_distance_ratio_to_psf_sigma": 3.0,   // Separation in units of sigma
    "sep_image_count": 50,
    "sep_intensity_prefactor_to_bg_level": 10.0,
    "sep_psf_sigma": 1.5,
    "sep_img_width": 80,
    "sep_bg_level": 1000,
    "sep_random_seed": 0
}
```

### Analysis Fields

```json
{
    "analyze_the_dataset?": true,
    "ana_random_seed": 0,
    "ana_predefined_psf_sigma": 2.0,          // Known PSF width
    "ana_use_premature_hypothesis_choice?": false,  // Stop early if xi increases
    "ana_maximum_hypothesis_index": 5,        // Test up to 5 particles
    "ana_timeout_per_image": 120,             // Seconds (optional, default 3600)
    "ana_delete_the_dataset_after_analysis?": false
}
```

### Optional Fields

- `"file_format": "png"` (default: `"tiff"`)
- `"ana_timeout_per_image"`: Maximum seconds per image (default 3600)

## Output Files

### Generated Images

**Location:** `./datasets/{image_folder_namebase}/`

**Filenames:**
- Regular dataset: `count{n}_index{i}.tiff` (n = particle count, i = image index)
- Separation test: `separation_psf{sigma}_sep{distance}_index{i}.tiff`
- Config used: `config_used.json`

### Analysis Results

**Location:** `./analyses/{image_folder_namebase}_code_ver{date}/`

**Files:**
1. **`{prefix}_config_used.json`**: Configuration that generated the results
2. **`{prefix}_label_prediction_log.csv`**: Per-image results
   - Columns: Input Image File, Actual Particle Count, Estimated Particle Count, Determined Particle Intensities
3. **`{prefix}_metrics_log.csv`**: Combined metrics from all hypothesis tests
   - Columns: image_filename (h number), true_count, h number, selected?, xi, lli, penalty, fisher_info, fit_parameters, xi_aic, xi_bic, penalty_aic, penalty_bic
4. **`{prefix}_confusion_mat.csv`**: Confusion matrix (rows = actual, cols = estimated). Not generated for experimental images without a ground truth.
5. **`{prefix}_confusion_mat.png`**: Heatmap visualization with performance metrics. Not generated for experimental images without a ground truth.
6. **`{prefix}_scores.csv`**: Summary statistics
   - Flat weight Accuracy, Within-One Accuracy, Overestimation Rate, Underestimation Rate, MAE, RMSE. Not generated for experimental images without a ground truth.

Where `{prefix}` = `{image_folder_namebase}_code_ver{code_version_date}`

## Command-Line Arguments

### main.py

```powershell
python main.py [-h] [-c CONFIG_DIR] [-p]

Arguments:
  -h, --help                Show help message
  -c, --config-files-dir    Directory containing config JSON files (default: ./configs)
  -p, --parallel            Enable parallel processing with ProcessPoolExecutor
```

### preprocess_exp_data.py

```powershell
python preprocess_exp_data.py [-h] [-t] [-f FOLDER] [-s SIZE] [-o OVERLAP] 
                               [-i INTERVAL] [-c CROP] [-m MAXHINDEX]
                               [--save-plots] [--predefined-sigma SIGMA]
                               [--config-subdir SUBDIR]

Arguments:
  -t, --terminal              Run without UI dialogs
  -f, --folder FOLDER         Path to TIFF images folder
  -s, --size SIZE             Sub-image size (pixels, square)
  -o, --overlap OVERLAP       Overlap between tiles (pixels, default 0)
  -i, --interval INTERVAL     File processing interval (0=auto, 1=all, N=every Nth)
  -c, --crop CROP             Crop fraction for raw images (default 0.7)
  -m, --maxhindex N           Max hypothesis index in config (default 5)
  --save-plots                Save plots to file instead of displaying
  --predefined-sigma SIGMA    Use fixed PSF sigma in config file (skip Gaussian fitting)
  --config-subdir SUBDIR      Subfolder under ./configs for output configs
```

## Algorithm Overview

The detection algorithm uses a generalized maximum likelihood test approach:

1. **Hypothesis Generation**: Test hypotheses H₀ (0 particles), H₁ (1 particle), ..., Hₙ (n particles)
2. **Maximum Likelihood Estimation**: For each hypothesis, optimize particle positions and intensities
3. **Model Selection**: Choose hypothesis that minimizes information criterion:
   - ξ = -log(L) + penalty
4. **Tiling Support**: Large images divided into overlapping tiles, detections merged

**Key Features:**
- PSF modeled as 2D Gaussian integrated over pixel areas
- Poisson noise model for photon counting
- Fisher information matrix for uncertainty quantification
- Coincident particle merging for tile-based analysis

## Performance Notes

- **Parallel processing** recommended for large datasets (use `--parallel` flag)
- **Timeout handling**: Images exceeding `ana_timeout_per_image` are skipped with logged warnings
- **Memory usage**: Large images automatically tiled (threshold: 160×160 pixels)
- **Typical runtime**: ~1-10 seconds per 100×100 image (depends on max hypothesis index and particle count)

---
