# How to run tests:

>python tests.py

## Command-line Options

You can customize the execution of the script using the following command-line options:

- `--foldername`: The name of the folder for the test images to be created and used. This is a required argument. Example: `--foldername test_images`

- `--last_h_index`: The last h index to be tested. Default is `3`. Example: `--last_h_index 3`

- `--rand_seed`: The random seed for image analysis. Default is `0`. Example: `--rand_seed 0`

- `--delete_images_folder`: Option to delete the folder containing the test images after the test is run. Default is `True`. Example: `--delete_images_folder True`

- `--num_images`: The number of images to be generated. Default is `2`. Example: `--num_images 2`

- `--amp_to_bgs`: The amplitude to background ratio. Default is `20`. Example: `--amp_to_bgs 20`

- `--normalized_amp_sd`: The normalized amplitude standard deviation (between 0 and 0.25). Default is `0.1`. Example: `--normalized_amp_sd 0.1`

- `--image_size`: The size of the generated images. Default is `20`. Example: `--image_size 20`

- `--background_level`: The background level of the generated images. Default is `500`. Example: `--background_level 500`

- `--psf_sd`: The standard deviation of the point spread function. Default is `1.39`. Example: `--psf_sd 1.39`

To run the script with these options, use the following format:

```bash
python tests.py --foldername test_images --last_h_index 3 --rand_seed 0 --delete_images_folder True --num_images 2 --amp_to_bgs 20 --normalized_amp_sd 0.1 --image_size 20 --background_level 500 --psf_sd 1.39
```


## Outputs

The script generates two types of CSV files in the `runs` folder:

1. **Test Metrics CSV**: This file contains the metrics calculated from the test. The file is named in the format `{input_image_file}-{current_date}-run{run_number}-randseed{rand_seed}_test_metrics.csv`. For example, if your input image file is named `image1`, you run the script on 2022-01-01, it's the first run of the day, and the random seed is 0, the file will be named `image1-2022-01-01-run1-randseed0_test_metrics.csv`.

2. **Fittings CSV**: This file contains the results of the fittings performed by the script. The file is named in the format `{input_image_file}-{current_date}-run{run_number}-randseed{rand_seed}_fittings.csv`. For example, for the same input image file, date, run number, and random seed as above, the file will be named `image1-2022-01-01-run1-randseed0_fittings.csv`.

Both files are saved in the `runs` folder in the same directory as the script. If the `runs` folder does not exist, the script will create it.

Please note that the `run_number` is automatically incremented for each run to avoid overwriting previous results. The `rand_seed` is a parameter that you can set when running the script.


In addition to the Test Metrics and Fittings CSV files, the script also generates a log file named `_actual_vs_counted.csv` in the `runs/{folder_name}` directory. This file logs the actual number of particles in each image and the number of particles estimated by the script.

The `_actual_vs_counted.csv` file has the following columns:

- `Actual Particle Number`: The actual number of particles in the image. This is extracted from the filename of the image.

- `Estimated Particle Number`: The number of particles estimated by the script. This is calculated by the script based on the image analysis.

This log file is useful for comparing the performance of the script in estimating the number of particles in different images. It can help you understand how accurate the script is and identify any images where the script's estimates are significantly different from the actual number of particles.

Here's an example of how the log file might look:

| Actual Particle Number | Estimated Particle Number |
|------------------------|---------------------------|
| 4                      | 3                         |
| 1                      | 1                         |
....
