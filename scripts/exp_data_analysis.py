# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict

# def plot_occurrence_count_analysis(folder_paths):
#     """
#     Analyzes occurrence count CSV files across multiple folders and plots mean ± SD for each.
    
#     Parameters:
#         folder_paths (list): List of folder paths to analyze
    
#     Returns:
#         dict: Dictionary containing processed data for each folder
#     """
#     # colors = plt.cm.tab10(np.linspace(0, 1, len(folder_paths)))  # Generate distinct colors
#     # colors = plt.cm.Blues(np.linspace(0.3, 1, len(folder_paths)))
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(folder_paths)))
#     # colors = plt.cm.plasma(np.linspace(0, 0.7, len(folder_paths)))

#     results = {}
    
#     plt.figure(figsize=(7, 4))
    
#     for i, folder_path in enumerate(folder_paths):
#         # Dictionary to store data from all subfolders for this folder
#         p_occurrence_data = defaultdict(list)
#         subfolders_processed = 0
        
#         print(f"\nProcessing folder: {folder_path}")
        
#         # Walk through all subfolders
#         for root, dirs, files in os.walk(folder_path):
#             for file in files:
#                 if file.endswith("occurence_count.csv"):
#                     file_path = os.path.join(root, file)
                    
#                     try:
#                         # Read the CSV file
#                         df = pd.read_csv(file_path)
                        
#                         # Ensure the required columns exist
#                         if 'Estimated Count' in df.columns and ('Occurrence' in df.columns or 'Frequency' in df.columns):
#                             all_occurrences = df['Occurrence'].sum() if 'Occurrence' in df.columns else df['Frequency'].sum()
#                             max_estimated_count = df['Estimated Count'].max()
#                             # Store data for each estimated count
#                             for row_index, row in df.iterrows():
#                                 estimated_count = int(row['Estimated Count'])
#                                 occurrence = int(row['Occurrence']) if 'Occurrence' in df.columns else int(row['Frequency'])
#                                 p_occurrence = occurrence / all_occurrences if all_occurrences > 0 else 0
#                                 p_occurrence_data[estimated_count].append(p_occurrence)

#                             for missed_count in range(max_estimated_count + 1, 6):
#                                 p_occurrence_data[missed_count].append(0)
                            
#                             subfolders_processed += 1
#                             print(f"  Processed: {file_path}")
#                         else:
#                             print(f"  Warning: Required columns not found in {file_path}")
                            
#                     except Exception as e:
#                         print(f"  Error reading {file_path}: {e}")
        

#         if subfolders_processed == 0:
#             print(f"  No valid occurrence_count.csv files found in {folder_path}!")
#             continue
        
#         print(f"  Processed {subfolders_processed} occurrence count files")
        
#         total_occurrences_per_est_count = np.array([sum(occurrences) for occurrences in occurrence_data.values()])        

#         estimated_counts = sorted(occurrence_data.keys())

#         for count in estimated_counts:
#             num = occurrence_data[count]
            
#             mean_count = np.mean(num)
#             std_count = np.std(num, ddof=1) if len(num) > 1 else 0

#             freq = mean_count / total_occurrences_per_est_count[count] if total_occurrences_per_est_count[count] > 0 else 0
#             std = std_count / total_occurrences_per_est_count[count] if total_occurrences_per_est_count[count] > 0 else 0
            
#             mean_counts.append(mean_count)
#             std_counts.append(std_count)

#             freq_means.append(freq)
#             freq_stds.append(std)        

#         # Store results for this folder
#         folder_name = os.path.basename(folder_path.rstrip('/\\'))
#         results[folder_name] = {
#             'estimated_counts': estimated_counts,
#             'means': mean_counts,
#             'stds': std_counts,
#             'freq_means': freq_means,
#             'freq_stds': freq_stds,
#             'subfolders_processed': subfolders_processed
#         }
        
#         # Plot mean with error bars (SD) for this folder
#         plt.errorbar(estimated_counts, freq_means, yerr=freq_stds,
#                 marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
#                 color=colors[i], label=folder_name, alpha=0.5)
        
#         print(f"  Plotted data for {folder_name}")
    
#     # Configure the plot
#     plt.xlabel('Estimated Count', fontsize=12)
#     plt.ylabel('Frequency', fontsize=12)
#     plt.grid(True, alpha=0.3)
    
#     # Set x-axis to show integer ticks
#     all_counts = set()
#     for result in results.values():
#         all_counts.update(result['estimated_counts'])
#     plt.xticks(sorted(all_counts))
    
#     # Add legend outside the plot area
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     # Adjust y-axis limits
#     all_max_values = []
#     for result in results.values():
#         # if result['means'] and result['stds']:
#         if result['freq_means'] and result['freq_stds']:
#             max_val = max([m + s for m, s in zip(result['freq_means'], result['freq_stds'])])
#             all_max_values.append(max_val)
    
#     if all_max_values:
#         y_max = max(all_max_values)
#         plt.ylim(0, y_max * 1.1)
    
#     plt.tight_layout()
#     plt.show()
#     pass
    
#     return results



import shutil
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

def plot_occurrence_count_analysis_improved(folder_paths):
    """
    Analyzes occurrence count CSV files using proper statistical methods for proportional data.
    
    Uses pooled proportions instead of averaging individual proportions to properly weight
    experiments with different sample sizes.
    
    Parameters:
        folder_paths (list): List of folder paths to analyze
    
    Returns:
        dict: Dictionary containing processed data for each folder
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, len(folder_paths)))
    results = {}
    
    plt.figure(figsize=(10, 6))
    
    for i, folder_path in enumerate(folder_paths):
        # Store raw data for proper statistical analysis
        occurrence_data = defaultdict(list)  # estimated_count -> list of occurrences
        total_data = []  # list of total occurrences per CSV
        csv_data = []  # list of (estimated_count -> occurrence) dicts per CSV
        
        subfolders_processed = 0
        
        print(f"\nProcessing folder: {folder_path}")
        
        # Collect all raw data first
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith("occurence_count.csv") or file.endswith("occurrence_count.csv"):
                    file_path = os.path.join(root, file)
                    
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Handle column name variations
                        occurrence_col = 'Occurrence' if 'Occurrence' in df.columns else 'Frequency'
                        
                        if 'Estimated Count' in df.columns and occurrence_col in df.columns:
                            # Calculate total for this CSV
                            total_occurrences = df[occurrence_col].sum()
                            total_data.append(total_occurrences)
                            
                            # Store individual CSV data
                            csv_dict = {}
                            for _, row in df.iterrows():
                                estimated_count = int(row['Estimated Count'])
                                occurrence = int(row[occurrence_col])
                                csv_dict[estimated_count] = occurrence
                                # Also collect for pooled analysis
                                occurrence_data[estimated_count].append(occurrence)
                            
                            csv_data.append(csv_dict)
                            subfolders_processed += 1
                            print(f"  Processed: {file_path}")
                        else:
                            print(f"  Warning: Required columns not found in {file_path}")
                            
                    except Exception as e:
                        print(f"  Error reading {file_path}: {e}")
        
        if subfolders_processed == 0:
            print(f"  No valid occurrence count files found in {folder_path}!")
            continue
        
        print(f"  Processed {subfolders_processed} CSV files")
        
        # Method 1: Pooled Proportions (Recommended)
        estimated_counts = sorted(occurrence_data.keys())
        pooled_occurrences = [sum(occurrence_data[count]) for count in estimated_counts]
        total_pooled = sum(pooled_occurrences)
        
        pooled_proportions = [occ / total_pooled for occ in pooled_occurrences]
        
        # Calculate confidence intervals for pooled proportions
        pooled_ci_lower = []
        pooled_ci_upper = []
        
        for occ in pooled_occurrences:
            if occ > 0 and total_pooled > occ:
                # Wilson score interval (better for proportions near 0 or 1)
                p = occ / total_pooled
                n = total_pooled
                z = 1.96  # 95% confidence interval
                
                denominator = 1 + z**2/n
                center = (p + z**2/(2*n)) / denominator
                margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
                
                ci_lower = max(0, center - margin)
                ci_upper = min(1, center + margin)
            else:
                ci_lower = 0
                ci_upper = 0
            
            pooled_ci_lower.append(ci_lower)
            pooled_ci_upper.append(ci_upper)
        
        # Method 2: Individual CSV Proportions (for comparison)
        individual_proportions = defaultdict(list)
        for csv_dict in csv_data:
            csv_total = sum(csv_dict.values())
            for count in estimated_counts:
                prop = csv_dict.get(count, 0) / csv_total if csv_total > 0 else 0
                individual_proportions[count].append(prop)
        
        individual_means = [np.mean(individual_proportions[count]) for count in estimated_counts]
        individual_stds = [np.std(individual_proportions[count], ddof=1) if len(individual_proportions[count]) > 1 else 0 
                          for count in estimated_counts]
        
        # Store results
        folder_name = os.path.basename(folder_path.rstrip('/\\'))
        results[folder_name] = {
            'estimated_counts': estimated_counts,
            'pooled_proportions': pooled_proportions,
            'pooled_ci_lower': pooled_ci_lower,
            'pooled_ci_upper': pooled_ci_upper,
            'individual_means': individual_means,
            'individual_stds': individual_stds,
            'total_samples': total_pooled,
            'num_csvs': subfolders_processed
        }
        
        # Plot pooled proportions with confidence intervals
        yerr_lower = [p - ci_l for p, ci_l in zip(pooled_proportions, pooled_ci_lower)]
        yerr_upper = [ci_u - p for p, ci_u in zip(pooled_proportions, pooled_ci_upper)]
        
        plt.errorbar(estimated_counts, pooled_proportions, 
                    yerr=[yerr_lower, yerr_upper],
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
                    color=colors[i], label=f'{folder_name} (n={total_pooled})', alpha=0.8)
        
        print(f"  Plotted pooled data for {folder_name} (total samples: {total_pooled})")
    
    # Configure the plot
    plt.xlabel('Estimated Count', fontsize=12, fontweight='bold')
    plt.ylabel('Proportion of Total Occurrences', fontsize=12, fontweight='bold')
    plt.title('Occurrence Count Analysis: Pooled Proportions with 95% CI', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show integer ticks
    all_counts = set()
    for result in results.values():
        all_counts.update(result['estimated_counts'])
    plt.xticks(sorted(all_counts))
    
    # Set y-axis to log scale if data spans multiple orders of magnitude
    y_values = []
    for result in results.values():
        y_values.extend(result['pooled_proportions'])
    
    if max(y_values) / min([y for y in y_values if y > 0]) > 100:
        # plt.yscale('log')
        plt.ylabel('Proportion of Total Occurrences (log scale)', fontsize=12, fontweight='bold')
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for folder_name, data in results.items():
        print(f"\n{folder_name}:")
        print(f"  Total samples: {data['total_samples']:,}")
        print(f"  Number of CSV files: {data['num_csvs']}")
        print("  Estimated Count -> Pooled Proportion (95% CI):")
        for i, count in enumerate(data['estimated_counts']):
            prop = data['pooled_proportions'][i]
            ci_l = data['pooled_ci_lower'][i]
            ci_u = data['pooled_ci_upper'][i]
            print(f"    {count}: {prop:.6f} ({ci_l:.6f} - {ci_u:.6f})")
    
    return results


def plot_occurrence_count_analysis_sd_methods(folder_paths, plot_title="Occurrence Count Analysis: Weighted SD Method"):
    """
    Analyzes occurrence count CSV files using different SD-based methods for error bars.
    
    Provides three SD-based approaches:
    1. Weighted SD (recommended)
    2. Binomial SD (theoretical)
    3. Bootstrap SD (resampling-based)
    
    Parameters:
        folder_paths (list): List of folder paths to analyze
        plot_title (str): Title of the plot
    
    Returns:
        dict: Dictionary containing processed data for each fold  expr
    """
    colors = plt.cm.turbo(np.linspace(0, 1, len(folder_paths)+1))
    results = {}
    
    plt.figure(figsize=(8, 5))
    
    for folder_index, folder_path in enumerate(folder_paths):
        # Store raw data for analysis
        csv_data = []  # list of (estimated_count -> occurrence, total) tuples
        occurrence_data = defaultdict(list)  # estimated_count -> list of occurrences
        
        subfolders_processed = 0
        
        print(f"\nProcessing folder: {folder_path}")
        
        # Collect all raw data
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith("occurence_count.csv") or file.endswith("occurrence_count.csv"):
                    file_path = os.path.join(root, file)
                    
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Handle column name variations
                        occurrence_col = 'Occurrence' if 'Occurrence' in df.columns else 'Frequency'
                        
                        if 'Estimated Count' in df.columns and occurrence_col in df.columns:
                            # Calculate total for this CSV
                            total_occurrences = df[occurrence_col].sum()
                            
                            # Store individual CSV data
                            csv_dict = {}
                            for _, row in df.iterrows():
                                estimated_count = int(row['Estimated Count'])
                                occurrence = int(row[occurrence_col])
                                csv_dict[estimated_count] = occurrence
                                occurrence_data[estimated_count].append(occurrence)
                            
                            csv_data.append((csv_dict, total_occurrences))
                            subfolders_processed += 1
                            print(f"  Processed: {file_path}")
                        else:
                            print(f"  Warning: Required columns not found in {file_path}")
                            
                    except Exception as e:
                        print(f"  Error reading {file_path}: {e}")
        
        if subfolders_processed == 0:
            print(f"  No valid occurrence count files found in {folder_path}!")
            continue
        
        print(f"  Processed {subfolders_processed} CSV files")
        
        # Get all estimated counts
        estimated_counts = sorted(occurrence_data.keys())
        
        # METHOD 1: Weighted Standard Deviation (Recommended)
        weighted_means = []
        weighted_sds = []
        
        for count in estimated_counts:
            # Calculate individual proportions and weights (sample sizes)
            proportions = []
            weights = []
            
            for csv_dict, total in csv_data:
                prop = csv_dict.get(count, 0) / total if total > 0 else 0
                proportions.append(prop)
                weights.append(total)
            
            proportions = np.array(proportions)
            weights = np.array(weights)
            
            # Weighted mean
            if np.sum(weights) > 0:
                weighted_mean = np.average(proportions, weights=weights)
                
                # Weighted standard deviation
                weighted_variance = np.average((proportions - weighted_mean)**2, weights=weights)
                weighted_sd = np.sqrt(weighted_variance)
            else:
                weighted_mean = 0
                weighted_sd = 0
            
            weighted_means.append(weighted_mean)
            weighted_sds.append(weighted_sd)
        
        # METHOD 2: Binomial Standard Deviation (Theoretical)
        # Based on pooled data but using binomial distribution formula
        pooled_occurrences = [sum(occurrence_data[count]) for count in estimated_counts]
        total_pooled = sum(pooled_occurrences)
        pooled_proportions = [occ / total_pooled for occ in pooled_occurrences]
        
        binomial_sds = []
        for i, count in enumerate(estimated_counts):
            p = pooled_proportions[i]
            n = total_pooled
            # Binomial SD: sqrt(n * p * (1-p)) / n = sqrt(p * (1-p) / n)
            binomial_sd = np.sqrt(p * (1 - p) / n) if n > 0 and p > 0 else 0
            binomial_sds.append(binomial_sd)
        
        # METHOD 3: Bootstrap Standard Deviation
        bootstrap_means = []
        bootstrap_sds = []
        n_bootstrap = 1000
        
        for count in estimated_counts:
            bootstrap_props = []
            
            # Resample CSVs with replacement
            for _ in range(n_bootstrap):
                sampled_indices = np.random.choice(len(csv_data), size=len(csv_data), replace=True)
                total_occ = 0
                count_occ = 0
                
                for idx in sampled_indices:
                    csv_dict, total = csv_data[idx]
                    count_occ += csv_dict.get(count, 0)
                    total_occ += total
                
                prop = count_occ / total_occ if total_occ > 0 else 0
                bootstrap_props.append(prop)
            
            bootstrap_means.append(np.mean(bootstrap_props))
            bootstrap_sds.append(np.std(bootstrap_props))
        
        # Store results
        folder_name = os.path.basename(folder_path.rstrip('/\\'))
        results[folder_name] = {
            'estimated_counts': estimated_counts,
            'weighted_means': weighted_means,
            'weighted_sds': weighted_sds,
            'pooled_proportions': pooled_proportions,
            'binomial_sds': binomial_sds,
            'bootstrap_means': bootstrap_means,
            'bootstrap_sds': bootstrap_sds,
            'total_samples': total_pooled,
            'num_csvs': subfolders_processed
        }
        
        # Plot weighted method (recommended)
        def replace_p_with_dot(folder_name):
            def replace(match):
                return '.'
            
            pattern = re.compile(r'(?<=\d)p(?=\d)|(?<= )p(?=\d)|^p(?=\d)')
            folder_name = pattern.sub(replace, folder_name)
            return folder_name

        folder_name_display = replace_p_with_dot(folder_name)
        plt.errorbar(estimated_counts, weighted_means, yerr=weighted_sds,
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
                color=colors[folder_index], label=f'{folder_name_display} (n={total_pooled})', alpha=0.6)
        
        print(f"  Plotted weighted SD data for {folder_name}")
    
    # Configure the plot
    plt.xlabel('Estimated Count', fontsize=12, fontweight='bold')
    plt.ylabel('Proportion of Total Occurrences', fontsize=12, fontweight='bold')
    plt.title(plot_title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show integer ticks
    all_counts = set()
    for result in results.values():
        all_counts.update(result['estimated_counts'])
    plt.xticks(sorted(all_counts))
    
    # Check if log scale is appropriate
    y_values = []
    for result in results.values():
        y_values.extend(result['weighted_means'])
    
    if max(y_values) / min([y for y in y_values if y > 0]) > 100:
        plt.yscale('log')
        plt.ylabel('Proportion (log scale)', fontsize=12, fontweight='bold')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Print comparison of methods
    print("\n" + "="*80)
    print("COMPARISON OF SD METHODS")
    print("="*80)
    for folder_name, data in results.items():
        print(f"\n{folder_name}:")
        print(f"  Total samples: {data['total_samples']:,}")
        print(f"  Number of CSV files: {data['num_csvs']}")
        print("\n  Method Comparison (Mean ± SD):")
        print("  Count | Weighted SD    | Binomial SD    | Bootstrap SD")
        print("  ------|----------------|----------------|----------------")
        
        for i, count in enumerate(data['estimated_counts']):
            w_mean, w_sd = data['weighted_means'][i], data['weighted_sds'][i]
            p_prop, b_sd = data['pooled_proportions'][i], data['binomial_sds'][i]
            bt_mean, bt_sd = data['bootstrap_means'][i], data['bootstrap_sds'][i]
            
            print(f"  {count:4d}  | {w_mean:.5f}±{w_sd:.5f} | {p_prop:.5f}±{b_sd:.5f} | {bt_mean:.5f}±{bt_sd:.5f}")
    
    return results


def compare_methods_visualization(folder_paths):
    """
    Creates a comparison plot showing both pooled and individual averaging methods.
    """
    results = plot_occurrence_count_analysis_improved(folder_paths)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
    
    for i, (folder_name, data) in enumerate(results.items()):
        counts = data['estimated_counts']
        
        # Plot pooled method
        ax1.errorbar(counts, data['pooled_proportions'],
                    yerr=[
                        [p - ci_l for p, ci_l in zip(data['pooled_proportions'], data['pooled_ci_lower'])],
                        [ci_u - p for p, ci_u in zip(data['pooled_proportions'], data['pooled_ci_upper'])]
                    ],
                    marker='o', label=folder_name, color=colors[i], alpha=0.8)
        
        # Plot individual averaging method
        ax2.errorbar(counts, data['individual_means'],
                    yerr=data['individual_stds'],
                    marker='s', label=folder_name, color=colors[i], alpha=0.8)
    
    ax1.set_title('Pooled Proportions Method\n(Recommended)', fontweight='bold')
    ax1.set_xlabel('Estimated Count')
    ax1.set_ylabel('Proportion')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title('Individual Averaging Method\n(For Comparison)', fontweight='bold')
    ax2.set_xlabel('Estimated Count')
    ax2.set_ylabel('Mean Proportion')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage:
folder_paths = [
    "./liu2024-s50/AuOnly",
    "./liu2024-s50/AgOnly",
    # "./liu2024-s50/AuAgCtrl", 
    "./liu2024-s50/AuAg1pM",
    "./liu2024-s50/AuAg5pM",
    "./liu2024-s50/AuAg50pM"
]

# # Example usage:
# folder_paths = [
#     "./liu2024-s100/AuOnly",
#     "./liu2024-s100/AgOnly",
#     "./liu2024-s100/AuAgCtrl", 
#     "./liu2024-s100/AuAg1pM",
#     "./liu2024-s100/AuAg5pM",
#     "./liu2024-s100/AuAg50pM"
# ]

# plot_occurrence_count_analysis_improved(folder_paths)
# plot_occurrence_count_analysis_sd_methods(folder_paths)

def organize_folders_by_tag(parent_dir):
    # Pattern to capture the tag
    pattern = re.compile(r"covid(?:_target|_virus)_(ctrl|control|2p5ul|5ul|10ul)")

    # Loop through subdirectories in the parent directory
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        if os.path.isdir(folder_path):
            match = pattern.search(folder_name)
            if match:
                tag = match.group(1)
                tag = "ctrl" if tag == "control" else tag
                # Destination directory
                dest_dir = os.path.join(parent_dir, tag)
                os.makedirs(dest_dir, exist_ok=True)

                # Move folder to destination
                dest_path = os.path.join(dest_dir, folder_name)
                if not os.path.exists(dest_path):
                    shutil.move(folder_path, dest_dir)
                    print(f"Moved: {folder_name} → {dest_dir}")
                else:
                    print(f"Skipped (already exists): {folder_name}")

if __name__ == "__main__":
    # current_folder = r"C:\github_repos\Hypothesis-Test-Based-Particle-Detection\liu_ns_1\171224_ns"
    # organize_folders_by_tag(current_folder)
    folder_paths = [
        r".\liu_ns_1\171224_ns\ctrl",
        r".\liu_ns_1\171224_ns\2p5ul",
        r".\liu_ns_1\171224_ns\5ul",
        r".\liu_ns_1\171224_ns\10ul",
    ]
    plot_occurrence_count_analysis_sd_methods(folder_paths, plot_title="171224 nanoshell exp")
