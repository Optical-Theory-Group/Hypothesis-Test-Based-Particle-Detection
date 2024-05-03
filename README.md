# Particle Counting via Hypothesis Testing

## Description

- This project generates and runs simulated nanoparticle assay images and perform hypothesis testing to count the number of particles in the image using detection theory. 
- Currently, it is using GMLR (generalized maximum likelihood rule) to decide the number of particles.

## How to run

- Type the following command

>python tests.py --config-file-folder "config file folder name"

or

>python tests.py -c config file folder name"

## Config file format example (.json)

{
    "dataset_name": "Uniform Intensity",
    "generate_dataset": true,
    "gen_randseed": 0,
    "gen_n_img_per_count": 200,
    "gen_mean_area_per_particle": 100,
    "gen_psf_sd": 1.4,
    "gen_img_width": 20,
    "gen_bg_level": 500,
    "gen_particle_int_to_bg_level": 20,
    "gen_particle_int_sd_to_mean_int": 0.0,
    "generated_img_folder_removal_after_counting": false,
    "analyze_dataset": true,
    "analysis_name": "r20240425",
    "analysis_randseed": 0,
    "analysis_psf_sd": 1.4,
    "analysis_use_exit_condition": false,
    "analysis_max_h_number": 5
}


## Outputs

### Generated Images

- The script will generate, if dictated by the config file, dataset images into image_dataset\{dataset_name}\.
- dataset_name is set inside the config file used. 
- File name: img{index}_{actual_number_of_particles}particles.tiff

### Analysis Results

- The script will generate, if dictated by the config file, analysis results inside runs\{dataset_name}_{run_name}\.
- dataset_name and run_name are set inside the config file used. 

#### Confusion Matrix

- The confusion matrix (actual number of particles vs estimated number of particles) is built from the test results and saved inside runs\{dataset_name}_{run_name}\.
- File name: actual_vs_counted.csv
- The file has the following columns:
1. `Actual Particle Number`: The actual number of particles in the image. This is extracted from the filename of the image.
2. `Estimated Particle Number`: The number of particles estimated by the script. This is calculated by the script based on the image analysis.

This log file is useful for comparing the performance of the script in estimating the number of particles in different images. It can help you understand how accurate the script is and identify any images where the script's estimates are significantly different from the actual number of particles.

Here's an example of how the log file might look:

| Actual Particle Number | Estimated Particle Number |
|------------------------|---------------------------|
| 4                      | 3                         |
| 1                      | 1                         |
....

#### Configuration Used

- The configuration used for the run is saved inside runs\{dataset_name}_{run_name}\.
- File name: config_used.json

#### Test Scores

- The test scores (Xi, Loglikelihood, Penalty metric) are saved in runs\{dataset_name}_{run_name}\image_log\.
- File name: {image_name}_scores.csv

#### Fit Results

- The fitting results (parameters) are saved in runs\{dataset_name}_{run_name}\image_log\.
- File name: {image_name}_fittings.csv
