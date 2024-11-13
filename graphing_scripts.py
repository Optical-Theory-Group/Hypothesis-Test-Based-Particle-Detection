import re
from matplotlib.colors import ListedColormap
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import os

# Function to calculate accuracy from a confusion matrix
def calculate_avg_accuracy(conf_matrix):
    correct_predictions = np.trace(conf_matrix)
    total_predictions = np.sum(conf_matrix)
    accuracy = correct_predictions / total_predictions
    return accuracy

# Load all CSV files in the current folder and calculate average accuracy for each
def calculate_avg_accuracies(folder_path):
    avg_accuracies = []
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder_path in subfolders:
        for filename in os.listdir(subfolder_path):
            if filename.endswith("confusion_mat.csv"):
                # Load the confusion matrix from CSV
                filepath = os.path.join(subfolder_path, filename)
                conf_matrix = pd.read_csv(filepath, header=None).values
                
                # Calculate accuracy and store it
                avg_accuracy = calculate_avg_accuracy(conf_matrix[1:,:])
                avg_accuracies.append((filename, avg_accuracy))
    
    return avg_accuracies

def calculate_weighted_accuracies(folder_path, lam):
    """ Calculate weighted accuracies for each CSV file in the folder. """
    weighted_accuracies = []
    
    # Calculate Poisson weights for counts 0 to 4 based on expected count
    poisson_weights = {k: poisson.pmf(k, lam) for k in range(5)}
    
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder_path in subfolders:
        for filename in os.listdir(subfolder_path):
            if filename.endswith("confusion_mat.csv"):
                # Load the confusion matrix from CSV
                filepath = os.path.join(subfolder_path, filename)
                conf_matrix = pd.read_csv(filepath, header=None).values
                conf_matrix = conf_matrix[1:,:]
                
                # Calculate weighted accuracy
                weighted_accuracy = 0
                for actual_count in range(5):  # Only actual counts 0 to 4
                    row_total = conf_matrix[actual_count].sum()
                    if row_total > 0:
                        accuracy = conf_matrix[actual_count, actual_count] / row_total
                        weighted_accuracy += accuracy * poisson_weights[actual_count]
                
                # Store the filename and weighted accuracy
                weighted_accuracies.append((filename, weighted_accuracy))
        
    return weighted_accuracies

def calculate_msle(conf_matrix):
    msle_values = []
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if conf_matrix[i, j] != 0:
                msle = (np.log1p(i) - np.log1p(j)) ** 2
                msle_values.append(msle)
    return np.mean(msle_values)

def calculate_rmse(conf_matrix):
    rmse = 0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            rmse += (i - j) ** 2 * conf_matrix[i, j]
    return np.sqrt(rmse / np.sum(conf_matrix))

def calculate_flat_rmse(folder_path, order):
    """ Calculate flat RMSE for each CSV file in the folder. """
    rmse_dict = {}
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder_path in subfolders:
        for filename in os.listdir(subfolder_path):
            if filename.endswith("confusion_mat.csv"):
                # Load the confusion matrix from CSV
                filepath = os.path.join(subfolder_path, filename)
                conf_matrix = pd.read_csv(filepath, header=None).values
                conf_matrix = conf_matrix[1:,:]

                rmse = calculate_rmse(conf_matrix)

                basename = os.path.basename(filename)
                parts = basename.split('_')
                xnames = format_xname(parts)
                rmse_dict[xnames] = rmse

    # Reorder rmse_dict based on names
    ordered_rmse_dict = {key: rmse_dict[key] for key in order if key in rmse_dict}
    rmse_dict = ordered_rmse_dict

    return rmse_dict

def calculate_mae(conf_matrix):
    mae = 0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            mae += np.abs(i - j) * conf_matrix[i, j]
    return mae / np.sum(conf_matrix)

def calculate_weighted_rmse(folder_path, lam, order):
    weighted_rmse = 0
    
    # Calculate Poisson weights for counts 0 to 4 based on expected count
    poisson_weights = {k: poisson.pmf(k, lam) for k in range(5)}

    w_rmse_dict = {}
    
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder_path in subfolders:
        for filename in os.listdir(subfolder_path):
            if filename.endswith("confusion_mat.csv"):
                # Load the confusion matrix from CSV
                filepath = os.path.join(subfolder_path, filename)
                conf_matrix = pd.read_csv(filepath, header=None).values
                conf_matrix = conf_matrix[1:,:]
                
                # Calculate weighted RMSE
                weighted_rmse = 0
                for actual_count in range(5):  # Only actual counts 0 to 4
                    row_total = conf_matrix[actual_count].sum()
                    if row_total > 0:
                        rmse = np.sqrt(np.sum((np.arange(conf_matrix.shape[1]) - actual_count) ** 2 * conf_matrix[actual_count]) / row_total)
                        weighted_rmse += rmse * poisson_weights[actual_count]
                        # total_weight += poisson_weights[actual_count]
                # weighted_rmses.append((filename, weighted_rmse))
                basename = os.path.basename(filename)
                parts = basename.split('_')
                xnames = format_xname(parts)

                w_rmse_dict[xnames] = weighted_rmse
    # Reorder rmse_dict based on names
    ordered_rmse_dict = {key: w_rmse_dict[key] for key in order if key in w_rmse_dict}
    w_rmse_dict = ordered_rmse_dict

    return w_rmse_dict

# Function to format xnames
def format_xname(parts):
    # xname = parts[1]
    if 'baseline' in parts[1]:
        xname = '1x'
    elif 'o' in parts[1]:
        xname = parts[1].replace('o', '/') + parts[2]
        xname = xname.replace('sqrt2', 'sqrt(2)')
    else:
        xname = parts[1] + '_' + parts[2].split('x')[0] + 'x' if 'x' in parts[2] else parts[1] + 'x'
        xname = xname.replace('sqrt2', 'sqrt(2)')
        xname = xname.replace('_', '.')

    if xname.endswith('xx'):
        xname = xname[:-1]

    return xname

def plot_all_accuracies(all_accuracies, x_vals, xlabel):
    plt.figure()
    for lam, accuracies in all_accuracies.items():
        # accuracy_values = [accuracy for filename, accuracy in accuracies]
        accuracy_dict = {}
        for filename, accuracy in accuracies:
            basename = os.path.basename(filename)
            parts = basename.split('_')
            xnames = format_xname(parts)
            accuracy_dict[xnames] = accuracy
        accuracy_values = [accuracy_dict.get(x_val, 0) for x_val in x_vals]
        if lam == 'avg':
            color = 'black' 
            label = 'simple avg of accuracies from each count'
        else:
            color = None
            label = f'assuming avg count per area={lam}'

        plt.plot(x_vals, accuracy_values, label=label, marker='o', color=color)
    
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.title(f'Accuracy vs {xlabel}')
    plt.legend(fontsize='small')
    plt.show(block=False)
    pass

def plot_rmses(all_rmses, x_vals, xlabel):
    plt.figure()
    for key, rmses in all_rmses.items():
        if key == 'regular':
            color = 'black'
            label = 'regular RMSE'
        else:
            color = None
            label = f'assuming avg count per area={key}'
        # plt.plot(x_vals, rmses, label=label, marker='o', color=color)
        plt.plot(x_vals, rmses, label=label, marker='o', color=color)
    
    plt.xlabel(xlabel)
    plt.ylabel('RMSE')
    plt.title(f'Root Mean Square Error vs {xlabel}')
    plt.show(block=False)
    plt.legend(fontsize='small')
    pass

def calculate_relative_measures(folder_path, x_vals, xlabel):
    """
    Calculate and plot relative measures (Mean Squared Logarithmic Error, Root Mean Square Error, and Mean Absolute Error)
    from confusion matrices stored in CSV files within subfolders of a given folder path.
    Parameters:
    folder_path (str): Path to the main folder containing subfolders with confusion matrix CSV files.
    x_vals (list): List of x-axis values for plotting.
    xlabel (str): Label for the x-axis of the plot.
    Returns:
    None
    """
    MSLEs = {}
    RMSEs = {}
    MAEs = {}
    
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder_path in subfolders:
        for filename in os.listdir(subfolder_path):
            if filename.endswith("confusion_mat.csv"):
                # Load the confusion matrix from CSV
                filepath = os.path.join(subfolder_path, filename)
                conf_matrix = pd.read_csv(filepath, header=None).values
                conf_matrix = conf_matrix[1:,:] #########################################

                # Calculate relative error, sMAPE, and MSLE
                msle = calculate_msle(conf_matrix)
                rmse = calculate_rmse(conf_matrix)
                mae = calculate_mae(conf_matrix)

                basename = os.path.basename(filename)
                parts = basename.split('_')
                xnames = format_xname(parts)

                MSLEs[xnames] = msle
                RMSEs[xnames] = rmse
                MAEs[xnames] = mae
    
    # Define a new colormap
    colormap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(range(3)))))
    plt.figure()
    plt.plot(x_vals, [MSLEs[xval] for xval in x_vals], label='Mean Squared Logarithmic Error', marker='o', color=colormap(0))
    plt.plot(x_vals, [RMSEs[xval] for xval in x_vals], label='Root Mean Square Error', marker='o', color='darkgray')
    plt.plot(x_vals, [MAEs[xval] for xval in x_vals], label='Mean Absolute Error', marker='o', color=colormap(1))
    
    plt.xlabel(xlabel)
    plt.ylabel('Measure Value')
    plt.ylim(0, 1.2)
    plt.title(f'Relative Measures vs {xlabel}')
    plt.legend(fontsize='small')
    plt.show(block=False)
    pass
            

def calculate_avg_estimated_counts(conf_matrix):
    avg_estimated_counts = {}
    for actual_count in range(conf_matrix.shape[0]):
        total_count = np.sum(conf_matrix[actual_count, :])
        if total_count == 0:
            avg_estimated_counts[actual_count] = 0
        else:
            weighted_sum = np.sum([estimated_count * conf_matrix[actual_count, estimated_count] for estimated_count in range(conf_matrix.shape[1])])
            avg_estimated_counts[actual_count] = weighted_sum / total_count
    return avg_estimated_counts

def calculate_avg_estimate_per_count(folder_path, tags):
    avg_estimates = {} 

    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder_path in subfolders:
        for filename in os.listdir(subfolder_path):
            if filename.endswith("confusion_mat.csv"):
                # Load the confusion matrix from CSV
                filepath = os.path.join(subfolder_path, filename)
                conf_matrix = pd.read_csv(filepath, header=None).values
                conf_matrix = conf_matrix[1:,:]

                avg_estimated_counts = calculate_avg_estimated_counts(conf_matrix)

                basename = os.path.basename(filename)
                parts = basename.split('_')
                xnames = format_xname(parts)

                avg_estimates[xnames] = avg_estimated_counts

    # Plot the average estimated counts for each actual count up to count==4
    plt.figure()
    colormap = ListedColormap(plt.cm.turbo(np.linspace(0, 1, len(tags))))
    # colormap = plt.cm.get_cmap('turbo', len(tags))
    # for idx, (tag, (filename, avg_estimated_counts)) in enumerate(zip(tags, avg_estimates)):
    #     counts = [count for count in avg_estimated_counts.keys() if count <= 4]
    #     estimates = [avg_estimated_counts[count] for count in counts]
    #     plt.plot(counts, estimates, label=tag, marker='o', color=colormap(idx))
    for idx, tag in enumerate(tags):
        count_dict = avg_estimates[tag]
        counts = [count for count in count_dict.keys() if count <= 4]
        estimates = [count_dict[count] for count in counts]
        plt.plot(counts, estimates, label=tag, marker='o', color=colormap(idx))
    
    # Draw a dotted line y=x up to count==4
    plt.plot([0, 4], [0, 4], linestyle='--', color='gray', label='y=x')
    
    plt.xlabel('Actual Count')
    plt.ylabel('Average Estimated Count')
    plt.xticks(range(5))

    plt.title('Average Estimated Count vs Actual Count')
    plt.legend(fontsize='small')
    plt.show(block=False)

    return avg_estimates

# # Folder path containing the CSV files
# folder_path = './11112024-analyses/psfwidth_test'
# folder_path = './11112024-analyses/snr_test'
folder_path = './11112024-analyses/zoom_test'
# folder_path = './11112024-analyses/scatterstrength_test'
# folder_path = './11112024-analyses/background_test'

# label = "Background (Data 3)"
# label = "PSFwidth, peak pixel value kept constant (Data 3)"
# label = "Scatter Strength (Data 3)"
# label = "SNR Factor (Data 3)"
label = "Zoom Factor (Data 3)"

# x_vals = ['0.125x', '0.25x', '0.5x', '1x', '2x', '4x', '8x']
# x_vals = ['1/8x', '1/4x', '1/2x', '1x', '2x']
x_vals = ['1/2sqrt(2)x', '1/2x', '1/sqrt(2)x', '1x', 'sqrt(2)x', '2x', '2sqrt(2)x']

tags = x_vals

lams = [0.5, 1, 2]

# # # Calculate and print the accuracy for each CSV file
# avg_accuracies = calculate_avg_accuracies(folder_path)
# weighted_accuracies = [calculate_weighted_accuracies(folder_path, lam) for lam in lams]
# all_accuracies = {}
# all_accuracies['avg'] = avg_accuracies
# for lam, weighted_accuracy in zip(lams, weighted_accuracies):
#     all_accuracies[lam] = weighted_accuracy
# plot_all_accuracies(all_accuracies, x_vals, xlabel=label) 

calculate_relative_measures(folder_path, x_vals, xlabel=label)



# flat_rmse = calculate_flat_rmse(folder_path, order=x_vals)
# weighted_rmses = [calculate_weighted_rmse(folder_path, lam, order=x_vals) for lam in lams]
# all_rmses = {}
# all_rmses['regular'] = flat_rmse.values()
# for lam, weighted_rmse in zip(lams, weighted_rmses):
#     all_rmses[lam] = weighted_rmse.values()
# plot_rmses(all_rmses, x_vals, xlabel=label)
# # pass

# ret = calculate_avg_estimate_per_count(folder_path, tags)
# pass

# 

pass