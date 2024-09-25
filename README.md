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
{</br>
    "image_folder_namebase": "general_test",</br>
    "code_version_date": "",</br>
</br>
    "image_format":"png", // optional. Default is "tiff"
</br>
    "separation_test_image_generation?": false,</br>
    "sep_distance_ratio_to_psf_sigma": 5.3,</br>
    "sep_image_count": 24,</br>
    "sep_intensity_prefactor_to_bg_level": 5.0,</br>
    "sep_psf_sigma": 2.0,</br>
    "sep_img_width": 40,</br>
    "sep_bg_level": 500,</br>
    "sep_random_seed": 0,</br>
</br>
    "generate_regular_dataset?": true,</br>
    "gen_random_seed": 0,</br>
    "gen_total_image_count": 11,</br>
    "gen_minimum_particle_count": 0,</br>
    "gen_maximum_particle_count": 5,</br>
    "gen_psf_sigma": 1,</br>
    "gen_img_width": 26,</br>
    "gen_bg_level": 500,</br>
    "gen_intensity_prefactor_to_bg_level_ratio_min": 5,</br>
    "gen_intensity_prefactor_to_bg_level_ratio_max": 5,</br>
    "gen_intensity_prefactor_coefficient_of_variation": 0.0,</br>
</br>
    "analyze_the_dataset?": true,</br>
    "ana_random_seed": 0,</br>
    "ana_predefined_psf_sigma": 1,</br>
    "ana_use_premature_hypothesis_choice?": false,</br>
    "ana_maximum_hypothesis_index": 5,</br>
    "Ana_delete_the_dataset_after_analysis?": true</br>
}</br>

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
