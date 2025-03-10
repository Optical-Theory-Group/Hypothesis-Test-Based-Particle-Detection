import numpy as np
import os
import json
import random

def update_config_json_randomize_seeds(directory, change_dict):
    """ Update all JSON files in the given directory with the new values in change_dict.
    
    Parameters
    ----------
    directory : str
        Path to the directory containing the JSON files.
    change_dict : dict
        Dictionary containing the new values to update in the JSON files.
    """

    # List of JSON files in the directory
    json_files = [filename for filename in os.listdir(directory) if filename.endswith('.json')]
    
    # Initialize random seed
    random.seed(42) 

    # Iterate over all files in the directory
    for json_file in json_files:
        filepath = os.path.join(directory, json_file)
        
        # Load the JSON file
        with open(filepath, 'r') as file:
            data = json.load(file)

        original_namebase = data['image_folder_namebase']
        data['image_folder_namebase'] = original_namebase + '_psf4'

        for k, v in change_dict.items():
            if k in data:
                prev = data[k]
                data[k] = v
                print(f"Updated '{k}' in {json_file} from {prev} to {v}")
        
        # set data['sep_random_seed'] to a random integer between 0 and 2**16

        # set data['ana_random_seed'] to a random integer between 0 and 2**16
        data['ana_random_seed'] = random.randint(0, 2**16)
        data['gen_random_seed'] = random.randint(0, 2**16)
        data['sep_random_seed'] = random.randint(0, 2**16)

        # delete all fields and values in data where the key starts with 'gen'.
        # keys_to_delete = [k for k in data.keys() if k.startswith('gen')]
        # for k in keys_to_delete:
        #     del data[k]
        
        # Save the updated JSON file
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)

# # Directory containing the JSON files
# directory = './example_config_folder'
# directory = 'psf4_const_peak_height_bg_test'
directory = './configs_to_run_on_server'
# directory = './psf_tolerance_test_configs'
# directory = './configs'

# # New value for "sep_bg_level"
change_dict = {
    # 'sep_bg_level': 2048,
    # 'code_version_date': '2025-01-27',
    'gen_total_image_count': 50000,
    # 'gen_psf_sigma': 4.0,
    # 'ana_predefined_psf_sigma': 4.0,
    # 'gen_particle_intensity_mean': 20000,
}
# change_dict = {}

update_config_json_randomize_seeds(directory, change_dict)
pass

def add_string_to_all_jsons(directory, addstring):
    """ Add a string to the filename of all JSON files in the given directory.
    
    Parameters
    ----------
    directory : str
        Path to the directory containing the JSON files.
    addstring : str
        String to add to the filename of all JSON files.
    """

    # List of JSON files in the directory
    json_files = [filename for filename in os.listdir(directory) if filename.endswith('.json')]

    # Iterate over all files in the directory
    for json_file in json_files:
        base, ext = os.path.splitext(json_file)
        new_name = base + addstring + ext
        os.rename(os.path.join(directory, json_file), os.path.join(directory, new_name))
        print(f"Renamed '{json_file}' to '{new_name}'")

def remove_last_n_chars_from_all_jsons(directory, n_chars):
    """ Remove the last n characters from the filename of all JSON files in the given directory.
    
    Parameters
    ----------
    directory : str
        Path to the directory containing the JSON files.
    n_chars : int
        Number of characters to remove from the filename of all JSON files.
    """

    # List of JSON files in the directory
    json_files = [filename for filename in os.listdir(directory) if filename.endswith('.json')]

    # Iterate over all files in the directory
    for json_file in json_files:
        base, ext = os.path.splitext(json_file)
        new_name = base[:-n_chars] + ext
        os.rename(os.path.join(directory, json_file), os.path.join(directory, new_name))
        print(f"Renamed '{json_file}' to '{new_name}'")

# add_string_to_all_jsons(directory, '_psf4')
# directory = 'psf4_const_peak_height_bg_test'
# remove_last_n_chars_from_all_jsons(directory, len("_config_used"))