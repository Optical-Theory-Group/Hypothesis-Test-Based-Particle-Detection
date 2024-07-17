import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import re
import pandas as pd
from collections import defaultdict

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

def extract_counting_results(prefix="separation_test_psf0_5"):
	directory = './runs/'
	# Dictionary to store the count of each estimated particle count vs. separation
	counts = defaultdict(lambda: defaultdict(int))
	pattern = re.compile(r'_sep(\d+(\_\d+)?)')

	# Iterate through each folder in the directory that starts with the given prefix
	for foldername in os.listdir(directory):
		if foldername.startswith(prefix):
			# Extract the separation value from the folder name
			match = pattern.search(foldername)			    
			separation = match.group(1)
			folder_path = os.path.join(directory, foldername)

			for filename in os.listdir(folder_path):
				if filename.endswith('label_prediction_log.csv'):
					file_path = os.path.join(folder_path, filename)
					df = pd.read_csv(file_path)
					for est in df['Estimated Particle Count']:
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
	percentage_df.to_csv(f'{prefix}_particle_count_percentage_vs_separation.csv', index=False)
	print(percentage_df)
	
	# Set Seaborn style
	sns.set_theme(style="whitegrid")
	palette = sns.color_palette("husl", 6)  # Using 'husl' colormap with 6 distinct colors

	percentage_data = {'separation': separations}
	for est in range(6):
		percentage_data[f'estimation=={est}'] = [counts[est][sep] / total_counts[sep] * 100 if sep in counts[est] else 0 for sep in separations]

	percentage_df = pd.DataFrame(percentage_data)
	percentage_df.to_csv(f'{prefix}_particle_count_percentage_vs_separation.csv', index=False)
	print(percentage_df)

	# Plotting the count of each estimated particle count as a function of separation
	plt.figure(figsize=(12, 6))
	for estimation in range(6):
		plt.plot(separations, data[f'estimation=={estimation}'], label=f'estimation={estimation}', marker='o', color=palette[estimation])

	plt.xlabel('Separation/Psf', fontsize=14)
	plt.ylabel('Count', fontsize=14)
	plt.title(f'PSF {prefix.split("psf")[-1]} Particle Count vs Separation', fontsize=16)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	plt.show(block=False)
	plt.savefig(f'psf{prefix.split("psf")[-1]}_particle_count_vs_separation.png')

	# Plotting the percentage of each estimated particle count as a function of separation
	plt.figure(figsize=(12, 6))
	for estimation in range(6):
		plt.plot(separations, percentage_data[f'estimation=={estimation}'], label=f'estimation={estimation}', marker='o', color=palette[estimation])

	plt.xlabel('Separation/Psf', fontsize=14)
	plt.ylabel('Percentage', fontsize=14)
	plt.title(f'PSF {prefix.split("psf")[-1]} Particle Count Percentage vs Separation', fontsize=16)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
	plt.grid(True)
	plt.tight_layout()
	plt.show(block=False)
	plt.savefig(f'psf{prefix.split("psf")[-1]}_particle_count_percentage_vs_separation.png')
	  
	# # Print the results
	# print("Count of each Estimated particle count as a function of separation:")
	# for estimation in range(6):
	#     print(f"separation\t{estimation=}\t{estimation=}")
	#     for separation, occurence in counts[estimation].items():
	#         print(f"{separation}\t{occurence}\t{occurence/sum(counts[estimation].values()):.5f}")
		
	# for est, separation_dict in counts.items():
	#     print(f"Estimated Particle Count {est}:")
	#     for separation, count in separation_dict.items():
	#         print(f"{separation}\t{count}")
  
	# print("Percentage of each Estimated particle count as a function of separation:")
	# for est, separation_dict in counts.items():
	#     print(f"Estimated Particle Count {est}:")
	#     for separation, count in separation_dict.items():
	#         print(f"{separation}\t{count/sum(separation_dict.values())}")
	# print("Percentage of each estimated particle count as a function of separation:")
	# for i, row in enumerate(counts):
	#     print(f"Estimated Particle Count {i} Percentage: {row/sum(row)}")
  
# Example usage
extract_counting_results(prefix="separation_test_psf2_0")

# correct_json_files(directory = './config_sep/')
# toggle_analysis(psf = 0.5, directory = './config_sep/')
# delete_images(n_remain = 500)
pass