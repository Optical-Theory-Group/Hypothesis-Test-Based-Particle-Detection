import numpy as np
import os
import json
import random

def update_sep_bg_level(directory, change_dict):

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

        for k, v in change_dict.items():
            if k in data:
                data[k] = v
                print(f"Updated '{k}' in {json_file} to {v}")
        
        # set data['sep_random_seed'] to a random integer between 0 and 2**16
        data['sep_random_seed'] = random.randint(0, 2**16)

        # set data['ana_random_seed'] to a random integer between 0 and 2**16
        data['ana_random_seed'] = random.randint(0, 2**16)

        # delete all fields and values in data where the key starts with 'gen'.
        keys_to_delete = [k for k in data.keys() if k.startswith('gen')]
        for k in keys_to_delete:
            del data[k]
        
        # Save the updated JSON file
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)

# Directory containing the JSON files
directory = './configs_to_run_on_server'

# New value for "sep_bg_level"
change_dict = {
    'sep_bg_level': 2048,
    'code_version_date': '2024-11-12',
}

update_sep_bg_level(directory, change_dict)