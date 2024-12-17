import re
from matplotlib.colors import ListedColormap
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import os
from mpl_toolkits.mplot3d import Axes3D

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

def calculate_weighted_estimated_counts(folder_path, lam, order):
    w_est_count_dict = {}
    
    # Calculate Poisson weights for counts 0 to 4 based on expected count
    poisson_weights = {k: poisson.pmf(k, lam) for k in range(10)}
    print(f"{lam=}")

    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder_path in subfolders:
        for filename in os.listdir(subfolder_path):
            if filename.endswith("confusion_mat.csv"):
                # Load the confusion matrix from CSV
                filepath = os.path.join(subfolder_path, filename)
                conf_matrix = pd.read_csv(filepath, header=None).values
                conf_matrix = conf_matrix[1:,:]
    
                basename = os.path.basename(filename)
                parts = basename.split('_')
                xnames = format_xname(parts)

                weighted_sum = 0
                for actual_count in range(10):
                    if actual_count > 4: 
                        if poisson_weights[actual_count] > 1e-4:
                            weighted_sum += actual_count * poisson_weights[actual_count]
                    else:
                        total_count = np.sum(conf_matrix[actual_count, :])
                        if total_count > 0:
                            calc = np.sum([estimated_count * conf_matrix[actual_count, estimated_count] for estimated_count in range(conf_matrix.shape[1])]) / total_count
                            p_of_actual_count = poisson_weights[actual_count]
                            weighted_sum += calc * p_of_actual_count
                print(f"{xnames=}, {weighted_sum=}")
                w_est_count_dict[xnames] = weighted_sum

    # Reorder w_est_count_dict based on names
    ordered_w_est_count_dict = {key: w_est_count_dict[key] for key in order if key in w_est_count_dict}
    w_est_count_dict = ordered_w_est_count_dict
                
    return w_est_count_dict

    
def plot_p2to1_vs_lambda(r_value=3.6):
    # Define a very narrow range for lambda values
    lambda_values_narrow = np.linspace(0, 4/10000, 1000)  # range of lambda from 0 to 0.0004
    y_values_narrow = np.pi * np.sqrt(lambda_values_narrow) * np.special.erf(np.sqrt(np.pi) * r_value * np.sqrt(lambda_values_narrow))

    # Plot the function in the narrow range
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values_narrow, y_values_narrow, label=r"$\pi\sqrt{\lambda}\mathrm{erf}(\sqrt{\pi}R\sqrt{\lambda})$")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\pi\sqrt{\lambda}\mathrm{erf}(\sqrt{\pi}R\sqrt{\lambda})$")
    plt.title(r"Plot of $\pi\sqrt{\lambda}\mathrm{erf}(\sqrt{\pi}R\sqrt{\lambda})$ as a function of $\lambda$ (R = 3.6, $\lambda$ from 0 to 0.0004)")
    plt.legend()
    plt.grid(True)
    plt.show()

# # Folder path containing the CSV files
# folder_path = './11112024-analyses/psfwidth_test'
# folder_path = './11112024-analyses/snr_test'
# folder_path = './11112024-analyses/zoom_test'
# folder_path = './11112024-analyses/scatterstrength_test'
folder_path = './11112024-analyses/background_test'

label = "Background (Data 3)"
# label = "PSFwidth, peak pixel value kept constant (Data 3)"
# label = "Scatter Strength (Data 3)"
# label = "SNR Factor (Data 3)"
# label = "Zoom Factor (Data 3)"

x_vals = ['0.125x', '0.25x', '0.5x', '1x', '2x', '4x', '8x']
# x_vals = ['1/8x', '1/4x', '1/2x', '1x', '2x']
# x_vals = ['1/2sqrt(2)x', '1/2x', '1/sqrt(2)x', '1x', 'sqrt(2)x', '2x', '2sqrt(2)x']

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

# calculate_relative_measures(folder_path, x_vals, xlabel=label)
def plot_weighted_counts_against_lam(weighted_counts, lams, xlabel):
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    
    for lam, w in zip(lams, weighted_counts):
        label = f'assuming avg count per area={lam}'
        axs[0].plot(w.keys(), w.values(), label=label, marker='o')
        axs[0].axhline(y=lam, color='gray', linestyle='--', label=f'lambda={lam}')
        axs[1].semilogy(list(w.keys()), list(w.values()), label=label, marker='o')
        axs[1].axhline(y=lam, color='gray', linestyle='--', label=f'lambda={lam}')
    
    for ax in axs:
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Estimated Count')
        ax.legend(fontsize='small')
    
    axs[0].set_title('Estimated Count vs Actual')
    axs[1].set_title('Estimated Count vs Actual (Semilog)')
    
    plt.suptitle(f'Estimated Count vs Actual, if avg count per area follows Poisson distribution with different lambdas\nSince we don\'t have data for actual_counts > 4, we assume 100% accuracy for those low occurence (approximation)\n')
    plt.show(block=False)
    pass

# weighted_counts = [calculate_weighted_estimated_counts(folder_path, lam=lam, order=x_vals) for lam in lams]
# plot_weighted_counts_against_lam(weighted_counts, lams, xlabel=label)



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

def plot_3d_surface_with_peaks(x_range, y_range, background_level, peaks):
    """
    Plot a 3D surface with a flat background and multiple Gaussian peaks.
    
    Parameters:
    x_range (tuple): Range of x values (min, max).
    y_range (tuple): Range of y values (min, max).
    background_level (float): The level of the flat background surface.
    peaks (list of dict): List of dictionaries, each containing 'x', 'y', 'height', and 'sigma' for a peak.
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    x, y = np.meshgrid(x, y)
    z = np.full_like(x, background_level)
    
    for peak in peaks:
        x0, y0, height, sigma = peak['x'], peak['y'], peak['height'], peak['sigma']
        z += height * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    # fig = plt.figure(figsize=(10,10))
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='Greys')
    ax.set_xticks([])
    ax.set_yticks([])
    current_zlim = ax.get_zlim()
    # ax.set_zlim(current_zlim[-1] * -0.5, 1.5 * current_zlim[1])
    ax.set_zlim(current_zlim[-1] * -0.1, 2.5 * current_zlim[1])
    # ax.set_zlim(0, 1.5 * current_zlim[1])
    ax.set_zticks([])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show(block=False)
    pass

# # Example usage
# x_range = (0, 100)
# y_range = (0, 100)
# background_level = 32000
# peaks = [
#     # {'x': 40, 'y': 51, 'height': 20000*25, 'sigma': 2.0},
#     # {'x': 57, 'y': 63, 'height': 20000*25, 'sigma': 2.0},
# ]

# plot_3d_surface_with_peaks(x_range, y_range, background_level, peaks)

def plot_xi_values(xi_values):
    """
    Plot xi values against their indices.
    
    Parameters:
    xi_values (list or array): List or array of xi values to plot.
    """
    x_values = list(range(len(xi_values)))
    plt.figure(figsize=(5,4))
    plt.plot(x_values, xi_values, marker='o', color='black')
    plt.xlabel('Hypothesis', fontsize=12 * 1.5)
    plt.ylabel(r'$\xi$', fontsize=12 * 1.5, rotation=0, labelpad=20)
    plt.xticks(fontsize=12 * 1.5)
    plt.yticks([])
    plt.tight_layout()
    plt.show(block=False)
    pass


xi_values = [-66382.125,
-66291.35499,
-66204.44662,
-66214.60394,
-66224.66774,
-66232.63556,
]

plot_xi_values(xi_values)