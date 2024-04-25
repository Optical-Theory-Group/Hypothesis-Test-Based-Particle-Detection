# Particle Counting via Hypothesis Testing

## Description

- This project generates and runs simulated nanoparticle assay images and perform hypothesis testing to count the number of particles in the image using detection theory. 
- Currently, it is using GMLR (generalized maximum likelihood rule) to decide the number of particles.

## How to run

- Type the following command

>python tests.py

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
