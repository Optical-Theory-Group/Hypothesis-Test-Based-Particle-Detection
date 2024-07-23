import random
from main import generate_separation_test_images
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import re
import pandas as pd
from collections import defaultdict

pd.options.display.float_format = '{:.4f}'.format

def delete_images(n_remain):
	# Define the directory containing the folders
	directory = './image_dataset/'

	# Iterate through each folder in the directory that starts with "separation_test_psf"
	for foldername in os.listdir(directory):
		if foldername.startswith('separation_test_psf2'):
			folder_path = os.path.join(directory, foldername)
			
			# List all TIFF files in the folder
			tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff')]
			
			# Sort the files based on the index number in their filename
			tiff_files.sort(key=lambda x: int(re.search(r'index(\d+)', x).group(1)))
			
			# Keep the first 500 files and delete the rest
			for tiff_file in tiff_files[n_remain:]:
				os.remove(os.path.join(folder_path, tiff_file))
			print(f"{foldername} reduced.")
	print("Deletion completed.")
	
	
def toggle_analysis(psf, directory):
	
	# Iterate through each file in the directory
	for filename in os.listdir(directory):
		if filename.endswith('.json'):
			file_path = os.path.join(directory, filename)
			
			# Open and load the JSON file
			with open(file_path, 'r') as file:
				data = json.load(file)
			
			# Check if the analysis_predefined_psf_sd field is 0.5
			if data.get('analysis_predefined_psf_sd') == psf:
				# Set the analyze_the_dataset field to false
				if data['analyze_the_dataset'] == True:
					data['analyze_the_dataset'] = False
				else:
					data['analyze_the_dataset'] = True
				
				# Save the modified JSON back to the file
				with open(file_path, 'w') as file:
					json.dump(data, file, indent=4)
	print("Modifications completed.")
	
def correct_json_files(directory):
	# Iterate through each file in the directory
	for filename in os.listdir(directory):
		if filename.endswith('.json'):
			file_path = os.path.join(directory, filename)

			# Open and load the JSON file
			with open(file_path, 'r') as file:
				data = json.load(file)

			# Check if the image_folder_namebase field matches the incorrect format
			if 'image_folder_namebase' in data:
				namebase = data['image_folder_namebase']
				match = re.match(r'separation_test_psf(\d+)_sep(\d+_\d+)', namebase)
				if match:
					# Correct the format
					corrected_namebase = f'separation_test_psf{match.group(1)}_0_sep{match.group(2)}'
					data['image_folder_namebase'] = corrected_namebase

					# Save the corrected JSON back to the file
					with open(file_path, 'w') as file:
						json.dump(data, file, indent=4)
	print("Corrections completed.")

def process_separation_test_results(subdir='', prefix="separation_test_psf0_5", n_random_pick=None):
	""" Extracts the count of each estimated particle count as a function of separation from the log files in the given directory.
		Generates a CSV file containing the extracted data and plots the results.
	
		Args:
			prefix (str): The prefix of the folder name to search for in the directory.

		Returns:
			tuple: A tuple containing the probability of overlap per particle, probability of overlap per area, and surface densities.
	"""
	directory = os.path.join('./runs/', subdir)
	# Dictionary to store the count of each estimated particle count vs. separation
	counts = defaultdict(lambda: defaultdict(int))
	pattern = re.compile(r'_sep(\d+(\_\d+)?)')

	# Iterate through each folder in the directory that starts with the given prefix
	for foldername in os.listdir(directory):
		if foldername.startswith(prefix):
			# Extract the separation value from the folder name
			match = pattern.search(foldername)			    
			separation = match.group(1)
			separation = float(separation.replace('_', '.'))
			folder_path = os.path.join(directory, foldername)

			for filename in os.listdir(folder_path):
				if filename.endswith('label_prediction_log.csv'):
					file_path = os.path.join(folder_path, filename)
					df = pd.read_csv(file_path)

					if n_random_pick is not None:
						# Randomly pick n_random_pick elements from the DataFrame
						random_indices = random.sample(range(len(df)), n_random_pick)
						estimates = df['Estimated Particle Count'].iloc[random_indices]
					else:
						estimates = df['Estimated Particle Count']
					for est in estimates:
						if separation not in counts[est]:
							counts[est][separation] = 0
						counts[est][separation] += 1

	separations = sorted({sep for est_dict in counts.values() for sep in est_dict.keys()})

	data = {'separation': separations}

	for est in range(6):
		data[f'estimation=={est}'] = [counts[est][sep] if sep in counts[est] else 0 for sep in separations]
	
	df = pd.DataFrame(data)
	df.to_csv(f'{prefix}_particle_count_vs_separation.csv', index=False)
	print(df)

	total_counts = {sep: sum(counts[est][sep] for est in range(6)) for sep in separations}
	percentage_data = {'separation': separations}
	for est in range(6):
		percentage_data[f'estimation=={est}'] = [counts[est][sep]/total_counts[sep] if sep in counts[est] else 0 for sep in separations]

	percentage_df = pd.DataFrame(percentage_data)
	# percentage_df.to_csv(f'{prefix}_particle_count_percentage_vs_separation.csv', index=False)
	print(percentage_df)
	
	# Set Seaborn style
	sns.set_theme(style="whitegrid")
	palette = sns.color_palette("turbo", 6)  # Using 'viridis' colormap with 6 distinct colors

	percentage_data = {'separation': separations}
	for est in range(6):
		percentage_data[f'estimation=={est}'] = [counts[est][sep] / total_counts[sep] * 100 if sep in counts[est] else 0 for sep in separations]

	percentage_df = pd.DataFrame(percentage_data)
	# percentage_df.to_csv(f'{prefix}_particle_count_percentage_vs_separation.csv', index=False)
	print(percentage_df)

	# Plotting the count of each estimated particle count as a function of separation
	plt.figure(figsize=(12, 4))
	for estimation in range(6):
		plt.plot(separations, data[f'estimation=={estimation}'], label=f'estimation={estimation}', marker='o', color=palette[estimation])
	
	psf_float = float(prefix.split("psf")[-1].replace('_', '.'))

	plt.xlabel('Separation/Psf', fontsize=14)
	plt.xticks(fontsize=10)
	plt.ylabel('Count', fontsize=14)
	plt.title(f'PSF {prefix.split("psf")[-1]} Particle Count vs Separation', fontsize=16)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	plt.show(block=False)
	if n_random_pick is not None:
		plt.savefig(f'psf{prefix.split("psf")[-1]}_particle_count_vs_separation_randomly_picked.png')
	else:
		plt.savefig(f'psf{prefix.split("psf")[-1]}_particle_count_vs_separation.png')

	# Plotting the percentage of each estimated particle count as a function of separation
	plt.figure(figsize=(12, 4))
	for estimation in range(6):
		plt.plot(separations, percentage_data[f'estimation=={estimation}'], label=f'estimation={estimation}', marker='o', color=palette[estimation])

	plt.xlabel('Separation/Psf', fontsize=14)
	plt.ylabel('Percentage', fontsize=14)
	plt.title(f'PSF {prefix.split("psf")[-1]} Particle Count Percentage vs Separation', fontsize=16)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	# plt.show(block=False)
	if n_random_pick is not None:
		plt.savefig(f'psf{prefix.split("psf")[-1]}_particle_count_percentage_vs_separation_randomly_picked.png')
	else:
		plt.savefig(f'psf{prefix.split("psf")[-1]}_particle_count_percentage_vs_separation.png')

	surface_densities = np.linspace(0.0, 0.02, 100)
	p_est_1_per_particle = np.zeros(len(surface_densities))
	for j, eta in enumerate(surface_densities):
		prev_r = 0
		for i, r_over_psf in enumerate(separations):
			r = r_over_psf * psf_float
			dr = r - prev_r
			lam = eta * np.pi * r**2
			p_est_1_per_particle[j] += percentage_data['estimation==1'][i] / 100 * 2 * np.pi * lam * r * np.exp(-lam*np.pi*r**2) * dr
			prev_r = r
	
	plt.figure(figsize=(8, 4))
	plt.plot(surface_densities, p_est_1_per_particle, label='p_est_1_per_particle', color='blue')
	plt.xlabel('Surface Density (num of particle/pixel)', fontsize=14)
	plt.ylabel('p_est_1_per_particle', fontsize=14)
	plt.ylim(0, 0.6)
	plt.title(f'PSF{prefix.split("psf")[-1]} Probability of overlap (unresolvable particles) per particle vs Surface Density', fontsize=16)
	plt.legend(fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	# plt.show(block=False)
	if n_random_pick is not None:
		plt.savefig(f'psf{prefix.split("psf")[-1]}_p_est_1_per_particle(surface_density)_randomly_picked.png')
	else:
		plt.savefig(f'psf{prefix.split("psf")[-1]}_p_est_1_per_particle(surface_density).png')

	p_est_1_per_area = p_est_1_per_particle * surface_densities

	return p_est_1_per_particle, p_est_1_per_area, surface_densities

def plot_unresolv_prob_per_particle_vs_surface_density_for_all_psfs():

	p_est1_2_0, p_est1_2_0_per_area, surface_densities = process_separation_test_results(prefix="separation_test_psf2_0")
	p_est1_1_5, p_est1_1_5_per_area, _ = process_separation_test_results(prefix="separation_test_psf1_5")
	p_est1_1_0, p_est1_1_0_per_area, _ = process_separation_test_results(prefix="separation_test_psf1_0")
	p_est1_0_5, p_est1_0_5_per_area, _ = process_separation_test_results(prefix="separation_test_psf0_5")

	# Assuming p_est1_* are dictionaries with keys as separations and values as counts

	# Create a viridis palette
	palette = sns.color_palette("turbo", 4)

	# Plotting all p_est1's on a single plot
	plt.figure(figsize=(12, 6))

	plt.plot(surface_densities, p_est1_2_0, label='psf2_0', marker='o', markersize=3, color=palette[0])
	plt.plot(surface_densities, p_est1_1_5, label='psf1_5', marker='o', markersize=3, color=palette[1])
	plt.plot(surface_densities, p_est1_1_0, label='psf1_0', marker='o', markersize=3, color=palette[2])
	plt.plot(surface_densities, p_est1_0_5, label='psf0_5', marker='o', markersize=3, color=palette[3])

	plt.xlabel('Surface density (num_particle/pixel)', fontsize=14)
	plt.ylabel('Probability', fontsize=14)
	plt.title('Probability of overlap (unresolvable particles) per particle vs Surface Density', fontsize=16)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('p_overlap_per_particle_vs_surface_density_all_psfs.png')
	# plt.show(block=False)

	# Plotting all p_est1's on a single plot
	plt.figure(figsize=(12, 6))

	plt.plot(surface_densities, p_est1_2_0_per_area, label='psf2_0', marker='o', markersize=3, color=palette[0])
	plt.plot(surface_densities, p_est1_1_5_per_area, label='psf1_5', marker='o', markersize=3, color=palette[1])
	plt.plot(surface_densities, p_est1_1_0_per_area, label='psf1_0', marker='o', markersize=3, color=palette[2])
	plt.plot(surface_densities, p_est1_0_5_per_area, label='psf0_5', marker='o', markersize=3, color=palette[3])

	plt.xlabel('Surface density (num_particle/pixel)', fontsize=14)
	plt.ylabel('Probability', fontsize=14)
	plt.title('Expected number of overlaping pairs per Area', fontsize=16)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('expected_overlap_pairs_per_area_vs_surface_density_all_psfs.png')
	plt.show(block=False)

def create_config_files_for_separation_tests(ref_json_path='', dest_folder_path='./config_for_remote_exec/'):
	
	separations = np.arange(0.0, 7.0, 0.2)
	psfs = [0.5, 1.0, 1.5, 2.0, 3.0]
	for psf in psfs:
		for sep in separations:
			# Define the source and destination paths
			psf_str = f"{psf:.1f}".replace('.', '_')
			sep_str = f"{sep:.1f}".replace('.', '_')
			dest_path = os.path.join(dest_folder_path, f"psf{psf_str}_sep{sep_str}.json")
			# Read the JSON file
			if ref_json_path != '':
				with open(ref_json_path, 'r') as file:
					config_data = json.load(file)
			else:
				config_data = {}

			# set the following fields
			config_data['image_folder_namebase'] = f'psf{psf_str}_sep{sep_str}'
			config_data['code_version_date'] = '2024-07-19'

			# Set the fields for separation test image generation
			config_data['separation_test_image_generation?'] = True
			config_data['sep_psf_sd'] = psf
			config_data['sep_psf_ratio'] = round(sep, 2)
			config_data['sep_image_count'] = 7500
			config_data['sep_intensity_prefactor_to_bg_level'] = 5.0
			config_data['sep_img_width'] = 40
			config_data['sep_bg_level'] = 500
			config_data['sep_random_seed'] = 722 

			# Set the field for dataset generation to False
			config_data['generate_the_dataset?'] = False

			# Set the field for analysis to True and set the (pre-defined) psf 
			config_data['analyze_the_dataset?'] = False
			config_data['ana_predefined_psf_sd'] = psf
			config_data['ana_random_seed'] = 723	
			config_data['ana_use_premature_hypothesis_choice?'] = False
			config_data['ana_maximum_hypothesis_index'] = 5
			config_data['ana_delete_the_dataset_after_analysis?'] = True 

			# Save the modified JSON to the new file
			os.makedirs(os.path.dirname(dest_path), exist_ok=True)
			with open(dest_path, 'w') as file:
				json.dump(config_data, file, indent=4)
			print(f'Saved modified config to {dest_path}')

create_config_files_for_separation_tests(dest_folder_path='./config_sep_test_remote_2/')
# process_separation_test_results(subdir='220724', prefix="psf0_5", n_random_pick=500)
# process_separation_test_results(subdir='220724', prefix="psf1_0", n_random_pick=500)
# process_separation_test_results(subdir='220724', prefix="psf1_5", n_random_pick=500)
# process_separation_test_results(subdir='220724', prefix="psf2_0", n_random_pick=500)
# process_separation_test_results(subdir='220724', prefix="psf3_0", n_random_pick=500)
pass