from main import make_specific_images
from process_algorithms import generalized_maximum_likelihood_rule_on_rgb
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

def process_separation_test_results(subdir='', prefix="separation_test_psf0_5",):
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

					estimates = df['Estimated Particle Count']
					for est in estimates:
						if separation not in counts[est]:
							counts[est][separation] = 0
						counts[est][separation] += 1

	sep_to_psf_ratios = np.array(sorted({float(sep) for est_dict in counts.values() for sep in est_dict.keys()}))

	data = {'separation': sep_to_psf_ratios}

	for est in range(6):
		data[f'estimation=={est}'] = [counts[est][sep] if sep in counts[est] else 0 for sep in sep_to_psf_ratios]
	
	df = pd.DataFrame(data)
	df.to_csv(f'{prefix}_particle_count_vs_separation.csv', index=False)
	print(df)

	total_counts = {sep: sum(counts[est][sep] for est in range(6)) for sep in sep_to_psf_ratios}
	# Set Seaborn style

	sns.set_theme(style="whitegrid")
	palette = sns.color_palette("turbo", 6)  # Using 'viridis' colormap with 6 distinct colors

	percentage_data = {'separation': sep_to_psf_ratios}
	for est in range(6):
		percentage_data[f'estimation=={est}'] = [counts[est][sep] / total_counts[sep] * 100 if sep in counts[est] else 0 for sep in sep_to_psf_ratios]

	# percentage_df = pd.DataFrame(percentage_data)

	# # percentage_df.to_csv(f'{prefix}_particle_count_percentage_vs_separation.csv', index=False)
	# print(percentage_df)

	psf_float = float(prefix.split("psf")[-1].replace('_', '.'))

	# Plotting the count of each estimated particle count as a function of separation

	# _, axs = plt.subplots(2, 1, figsize=(12, 7))
	_, axs = plt.subplots(figsize=(12, 4))
	# plt.sca(axs[0]) # sca: set current axis
	for estimation in range(6):
		plt.plot(sep_to_psf_ratios, data[f'estimation=={estimation}'], label=f'estimation={estimation}', marker='o', color=palette[estimation])

	plt.xlabel('Separation/Psf (ratio)', fontsize=14)
	plt.xticks(fontsize=10)
	plt.ylabel('Count', fontsize=14)
	plt.title(f'PSF {prefix.split("psf")[-1]} Particle Count vs Separation', fontsize=16)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	plt.subplots_adjust(right=0.85)  # Adjust the right boundary of the plot to make space for the legend
	# plt.show(block=False)
	plt.savefig(f'psf{prefix.split("psf")[-1]}_particle_count_vs_separation_in_psf.png')

	_, axs = plt.subplots(figsize=(12, 4))
	# plt.sca(axs[1]) # sca: set current axis
	for estimation in range(6):
		plt.plot(sep_to_psf_ratios * psf_float, data[f'estimation=={estimation}'], label=f'estimation={estimation}', marker='o', markersize=1, color=palette[estimation])
	plt.xlabel('Separation (px)', fontsize=14)
	plt.xlim([-.5, 21.5])
	plt.xticks(fontsize=10)
	plt.ylabel('Count', fontsize=14)
	plt.title(f'PSF {prefix.split("psf")[-1]} Particle Count vs Separation', fontsize=16)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	plt.subplots_adjust(right=0.85)  # Adjust the right boundary of the plot to make space for the legend
	# plt.show(block=False)
	plt.savefig(f'psf{prefix.split("psf")[-1]}_particle_count_vs_separation_in_px.png')

	# surface_densities = np.array([0, 0.001, 0.01, 0.1, 1, 10])
	radiuses = sep_to_psf_ratios * psf_float
	pdf_w = np.zeros(len(radiuses))
	cdf_w = np.zeros(len(radiuses))
	
	# expected_num_of_unresolvably_overlapping_particles = np.zeros(len(radiuses))

	# Calculate the radius interval available
	dr = (radiuses[1] - radiuses[0]) * psf_float

	# for j, surf_den in enumerate(surface_densities):
	for i, radius in enumerate(radiuses):
		# Calculate the probability density function 
		pdf_w[i] = percentage_data['estimation==1'][i] / 100 * 2 * np.pi * radius

		# Calculate the cumulative density function
		if i == 0:
			cdf_w[i] = pdf_w[i] * dr
		else:
			cdf_w[i] = cdf_w[i-1] + pdf_w[i] * dr

	plt.figure(figsize=(8, 4))
	plt.plot(radiuses, cdf_w, color='red')
	plt.xlabel('Radius (px)', fontsize=14)
	plt.xlim([-.5, 21.5])
	plt.ylabel('Expected number', fontsize=14)
	plt.title(f'PSF {prefix.split("psf")[-1]} Expected num_particles unresolvably overlapping with an arbitrary particle / Surface Density', fontsize=12)
	plt.legend(fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	# plt.show(block=False)
	plt.savefig(f'psf{prefix.split("psf")[-1]}_expected_num_of_unresolvably_overlapping_particles per surface_density.png')
	pass


	# for surf_den in enumerate(surface_densities):

	# 	prev_r = 0
		
	# 	for i, r_over_psf in enumerate(sep_to_psf_ratios):
	# 		r = r_over_psf * psf_float
	# 		dr = r - prev_r
	# 		lam = surf_den * np.pi * r**2
	# 		p_est_1_per_particle[j] += percentage_data['estimation==1'][i] / 100 * 2 * np.pi * lam * r * np.exp(-lam*np.pi*r**2) * dr
	# 		prev_r = r
	
	# plt.close('all')
	# plt.figure(figsize=(8, 4))
	# plt.plot(surface_densities, p_est_1_per_particle, label='p_est_1_per_particle', color='blue')
	# plt.xlabel('Surface Density (num of particle/pixel)', fontsize=14)
	# plt.ylabel('p_est_1_per_particle', fontsize=14)
	# plt.ylim(0, 1.0)
	# plt.title(f'PSF{prefix.split("psf")[-1]} Probability of overlap (unresolvable particles) per particle vs Surface Density', fontsize=16)
	# plt.legend(fontsize=12)
	# plt.grid(True)
	# plt.tight_layout()
	# # plt.show(block=False)
	# plt.savefig(f'psf{prefix.split("psf")[-1]}_p_est_1_per_particle(surface_density).png')

	# p_est_1_per_area = p_est_1_per_particle * surface_densities

	# return cdf_w, p_est_1_per_area, surface_densities
	return cdf_w, radiuses, data[f'estimation==1']

def plot_estimate_1s():
	est_1 = {}
	radiuses = {} 
	psfs = [0.5, 1.0, 1.5, 2.0, 3.0]
	for psf in psfs:
		psf_string = f"{psf:.1f}".replace('.', '_')
		_, radiuses[psf_string], est_1[psf_string] = process_separation_test_results(subdir='220724', prefix=f"psf{psf_string}")

	palette = sns.color_palette("nipy_spectral", len(psfs))
	plt.figure(figsize=(8, 3))

	for psf in psfs:
		psf_string = f"{psf:.1f}".replace('.', '_')
		plt.plot(radiuses[psf_string], est_1[psf_string], label=f'psf{psf_string}', marker='x', markersize=3, color=palette[psfs.index(psf)])

	plt.xlabel('distance (px)', fontsize=12)
	plt.ylabel('Probability of estimating 2 as 1', fontsize=11)
	plt.title('Probability of estimating 2 as 1', fontsize=12)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('probability_of_estimating_2_as_1.png')
	plt.show(block=False)

def plot_unresolv_prob_per_particle_vs_radius_all_psfs():
	exp_num_overlap = {}
	radiuses = {}
	psfs = [0.5, 1.0, 1.5, 2.0, 3.0]
	for psf in psfs:
		psf_string = f"{psf:.1f}".replace('.', '_')
		exp_num_overlap[psf_string], radiuses[psf_string] = process_separation_test_results(subdir='220724', prefix=f"psf{psf_string}")

	palette = sns.color_palette("nipy_spectral", 5)
	plt.figure(figsize=(8, 3))

	for psf in psfs:
		psf_string = f"{psf:.1f}".replace('.', '_')
		plt.plot(radiuses[psf_string], exp_num_overlap[psf_string], label=f'psf{psf_string}', marker='x', markersize=3, color=palette[psfs.index(psf)])

	plt.xlabel('radius (px)', fontsize=12)
	plt.ylabel('Num_particle / Surf_density', fontsize=11)
	plt.title('Expected number of unresolvably overlapping particles with an arbitrary particle / Surface density', fontsize=12)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('expected_num_of_unresolvably_overlapping_particles_per_surface_density.png')
	plt.show(block=False)
	pass

def plot_unresolv_prob_per_particle_vs_surface_density_for_all_psfs():

	p_est1_2_0, p_est1_2_0_per_area, surface_densities = process_separation_test_results(subdir='220724', prefix="psf2_0")
	p_est1_1_5, p_est1_1_5_per_area, _ = process_separation_test_results(subdir='220724', prefix="psf1_5")
	p_est1_1_0, p_est1_1_0_per_area, _ = process_separation_test_results(subdir='220724', prefix="psf1_0")
	p_est1_0_5, p_est1_0_5_per_area, _ = process_separation_test_results(subdir='220724', prefix="psf0_5")
	p_est1_3_0, p_est1_3_0_per_area, _ = process_separation_test_results(subdir='220724', prefix="psf3_0")

	# Assuming p_est1_* are dictionaries with keys as separations and values as counts

	# Create a viridis palette
	palette = sns.color_palette("nipy_spectral", 5)

	# Plotting all p_est1's on a single plot
	plt.figure(figsize=(8, 3))

	plt.plot(surface_densities, p_est1_3_0, label='psf3_0', marker='x', markersize=3, color=palette[4])
	plt.plot(surface_densities, p_est1_2_0, label='psf2_0', marker='x', markersize=3, color=palette[0])
	plt.plot(surface_densities, p_est1_1_5, label='psf1_5', marker='x', markersize=3, color=palette[1])
	plt.plot(surface_densities, p_est1_1_0, label='psf1_0', marker='x', markersize=3, color=palette[2])
	plt.plot(surface_densities, p_est1_0_5, label='psf0_5', marker='x', markersize=3, color=palette[3])

	plt.xlabel('Surface density (num_particle/pixel)', fontsize=14)
	plt.ylabel('Probability', fontsize=14)
	plt.title('Probability of overlap (unresolvable particles) per particle vs Surface Density', fontsize=16)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('p_overlap_per_particle_vs_surface_density_all_psfs.png')
	# plt.show(block=False)

	# Plotting all p_est1's on a single plot
	plt.figure(figsize=(8, 3))

	plt.plot(surface_densities, p_est1_3_0_per_area, marker='x', markersize=3, label='psf3_0', color=palette[4])
	plt.plot(surface_densities, p_est1_2_0_per_area, marker='x', markersize=3, label='psf2_0', color=palette[0])
	plt.plot(surface_densities, p_est1_1_5_per_area, marker='x', markersize=3, label='psf1_5', color=palette[1])
	plt.plot(surface_densities, p_est1_1_0_per_area, marker='x', markersize=3, label='psf1_0', color=palette[2])
	plt.plot(surface_densities, p_est1_0_5_per_area, marker='x', markersize=3, label='psf0_5', color=palette[3])

	plt.xlabel('Surface density (num_particle/pixel)', fontsize=14)
	plt.ylabel('Probability', fontsize=14)
	plt.title('Expected number of overlaping pairs per Area', fontsize=16)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('expected_overlap_pairs_per_area_vs_surface_density_all_psfs.png')
	plt.show(block=False)

def create_config_files_for_separation_tests(ref_json_path='', dest_folder_path='./config_for_remote_exec/', psfs=[0.5, 1.0, 1.5, 2.0, 3.0]):
	
	separations = np.arange(0.0, 7.0, 0.2)
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
			config_data['code_version_date'] = '2024-07-31'

			# Set the fields for separation test image generation
			config_data['separation_test_image_generation?'] = True
			config_data['sep_psf_sd'] = psf
			config_data['sep_psf_ratio'] = round(sep, 2)
			config_data['sep_image_count'] = 10000
			config_data['sep_intensity_prefactor_to_bg_level'] = 5.0
			if psf == 3.0:
				config_data['sep_intensity_prefactor_to_bg_level'] = 20.0
			config_data['sep_img_width'] = 40
			config_data['sep_bg_level'] = 500
			config_data['sep_random_seed'] = 722 

			# Set the field for dataset generation to False
			config_data['generate_the_dataset?'] = False

			# Set the field for analysis to True and set the (pre-defined) psf 
			config_data['analyze_the_dataset?'] = True
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

# create_config_files_for_separation_tests(ref_json_path='config_sep_test_remote_2/psf3_0_sep6_8.json', dest_folder_path='./config_sep_test_scale_intensity/', psfs=[3.0])
# create_config_files_for_separation_tests(dest_folder_path='./config_sep_test_scale_intensity/', psfs=[3.0])
# create_config_files_for_separation_tests(dest_folder_path='./config_sep_test_random_offset_center/', psfs=[0.5])
# psfs = np.array([.5, 1, 1.5, 2, 3])
# psfs = np.array([0.5, 3])
# for psf in psfs:
# 	process_separation_test_results(subdir='2024-08-19', prefix=f"psf{psf}".replace('.', '_'))
# 	# process_separation_test_results(prefix=f"psf{psf}".replace('.', '_'))
# 	# plt.close('all')
# # pass
# # plot_unresolv_prob_per_particle_vs_surface_density_for_all_psfs()
# # plot_unresolv_prob_per_particle_vs_radius_all_psfs()
# # plot_estimate_1s()
# pass



img_param = [
            {'bg': [50, 40, 30], 'sz': 20, 'psf': 1},
            {'x': 5, 'y': 5, 'intensity': [1000, 100, 10]},
            {'x': 10, 'y': 10, 'intensity': [20, 2000, 200]},
            {'x': 15, 'y': 15, 'intensity': [25, 250, 2500]},
            ]
foldername = 'specific_images'
filename = 'count3-index0.tiff'
random_seed = 0
roi_image = make_specific_images(foldername, img_param, random_seed)
fit_results = generalized_maximum_likelihood_rule_on_rgb(roi_image, img_param[0]['psf'], last_h_index=5, random_seed=0, display_fit_results=True, display_xi_graph=False, use_exit_condi=False)
# res = analyze_image(os.path.join('./image_dataset/', foldername, filename), img_param[0]['psf'], 5, 0, './runs/specific_images', display_fit_results=True, display_xi_graph=True)
# print(f'Actual number of particles: {res["actual_num_particles"]}')
# print(f'Estimated number of particles: {res["estimated_num_particles"]}')
# print(f'Determined particle intensities: {res["determined_particle_intensities"]}')
# print(f'Metrics: {res["metrics"]}')
pass