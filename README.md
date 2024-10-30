# Hypothesis Test-Based Particle Detection

## Description

- This project generates and runs simulated nanoparticle assay images and perform hypothesis tests to detect particles in the image using information theory. 
- Currently, it is using GMLR (generalized maximum likelihood rule) to decide the number of particles.

## How to run

- Type the following command

>python tests.py -c config file folder name"

or

>python tests.py --config-file-folder "config file folder name"

## Config file format example (.json)
{</br>
    "image_folder_namebase": "general_test",</br>
    "code_version_date": "YYYY-MM-DD",</br>
</br>
    "file_format":"png", // optional. Default is "tiff"
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
    "gen_particle_intensity_mean": 2500,</br>
    "gen_particle_intensity_sd": 0,</br>
</br>
    "analyze_the_dataset?": true,</br>
    "ana_timeout_per_image?": 2 (optional),</br>
    "ana_random_seed": 0,</br>
    "ana_predefined_psf_sigma": 1,</br>
    "ana_use_premature_hypothesis_choice?": false,</br>
    "ana_maximum_hypothesis_index": 5,</br>
    "ana_delete_the_dataset_after_analysis?": true</br>
}</br>

## Outputs

### Generated Images

- The script will generate, if dictated by the config file, images into "./datasets/{image_folder_namebase}".
- {image_folder_namebase} is read from the config file used. 
- Inside the folder, images will be generated with name format "count{num_particles}_index{index_num}.tiff" if it is part of a regular dataset, and "count2_psf{psf_sigma}_index{index_num}.tiff" if it is part of a separation test dataset.
- config_used.json file will be also generated.

### Analysis Result

- The script will generate, if dictated by the config file, analysis results inside "analyses/{image_folder_namebase}_{code_version_date}/.
- image_folder_namebase and code_version_date are set inside the config file used.
- The following files will be created with the same prefix ({image_folder_namebase}_{code_version_date}):
    1. (prefix)_config_used.json (self explanatory file name)
    2. (prefix)_confusion_mat.csv (self explanatory file name)
    3. (prefix)_confusion_mat.png (self explanatory file name)
    4. (prefix)_label_prediction_log.csv (each line records the actual number of particles in the image (i.e., label) and the number algorithm determined (i.e., prediction))
    5. (prefix)_metric_log_per_image_hypothesis.csv (records all metrics for each hypothesis test done on each image)
    6. (prefix)_scores.csv (records the log likelihood, penalty, and the final xi scores)
