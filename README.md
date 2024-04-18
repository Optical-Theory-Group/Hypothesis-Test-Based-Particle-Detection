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
