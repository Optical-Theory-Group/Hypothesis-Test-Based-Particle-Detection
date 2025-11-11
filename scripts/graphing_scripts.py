import matplotlib as mpl 
mpl.rcParams['svg.fonttype'] = 'none'
import math
import re
from matplotlib.colors import ListedColormap
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import os
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import glob

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


def parse_xpart(chunk):
    """
    Convert strings like:
      '8x'              -> '8x'
      '1o_16x'          -> '1/16x'
      '1o_sqrt8x'       -> '1/2sqrt2x'
      '1o_sqrt4x'       -> '1/2x'
      'sqrt8x'          -> '2sqrt2x'
    and so on.
    """
    # 1) Check if it's '1o_' (meaning 1 over).
    is_over = False
    if chunk.startswith("1o_"):
        is_over = True
        chunk = chunk[3:]  # strip '1o_'

    # 2) Is it 'sqrt<number>x' or just '<number>x'?
    if chunk.startswith("baseline"):
        chunk = "1x"

    if chunk.startswith("sqrt"):
        # Parse the integer after 'sqrt'
        if chunk.startswith("sqrt") and chunk[-1].isdigit(): # adhoc fix of naming mistake.
            chunk += "x"
        m = re.match(r"^sqrt(\d+)x$", chunk)
        if not m:
            raise ValueError(f"Unexpected sqrt format in '{chunk}'")
        val = int(m.group(1))
        val_str = decompose_sqrt(val)  # e.g. sqrt8 -> '2sqrt2'
    else:
        # Otherwise, parse a pure integer, e.g. '16x'
        m = re.match(r"^(\d+)x$", chunk)
        if not m:
            raise ValueError(f"Unexpected format in '{chunk}'")
        val = int(m.group(1))
        val_str = str(val)

    # Attach "x"
    result = val_str + "x" if val_str != "1" else "1x"

    # If "1o_" was found, wrap it with "1/"
    if is_over:
        result = f"1/{result}"

    return result

def decompose_sqrt(num):
    """
    Convert 'sqrt<num>' into a form like:
      sqrt4  -> '2'
      sqrt8  -> '2sqrt2'
      sqrt16 -> '4'
      sqrt32 -> '4sqrt2'
      sqrt1  -> '1'
    """
    # Largest integer factor i where i*i <= num
    i = int(math.isqrt(num))  
    leftover = num // (i * i)

    if i == 1 and leftover == 1:
        # sqrt1 -> '1'
        return "1"
    if leftover == 1:
        # sqrt4 -> 2, sqrt16 -> 4, etc.
        return str(i)
    else:
        # e.g. sqrt8 -> i=2 leftover=2 => "2sqrt2"
        # e.g. sqrt32 -> i=5 leftover=? Actually i=5 means 25 leftover=32/25? 
        # Better approach: i=4 leftover=2 => "4sqrt2"
        # So let's ensure i = int(sqrt(num)) was correct. 
        # (math.isqrt(8) = 2 leftover = 2 => "2sqrt2")
        if i == 1:
            return f"sqrt{leftover}"  # e.g. sqrt2 -> 'sqrt2'
        return f"{i}sqrt{leftover}"

def format_xname(filename):
    """
    Extract the chunk after '-'.
    Example:  basename='d4-scatter-1o_sqrt8x_code_ver2024-11-29'
              pattern_part='d4-scatter-1o_sqrt8x'
              chunk='1o_sqrt8x'
              -> parse_xpart(...) -> '1/2sqrt2x'
    """
    basename = os.path.basename(filename)
    pattern_part = basename.split("_code_ver")[0]
    chunk = pattern_part.split("-")[-1]
    if chunk.startswith("analysis") and chunk.endswith("psf4"):
        if chunk.split("_")[2] == '1o':
            chunk = chunk.split("_")[2] + '_' + chunk.split("_")[3]
        else:
            chunk = chunk.split("_")[2] 
    else:
        if chunk.startswith("1o_"):
            chunk = f"{chunk.split("_")[0]}_{chunk.split("_")[1]}" 
            pass
        else:
            chunk = chunk.split("_")[0]

    return parse_xpart(chunk)


def plot_all_accuracies(all_accuracies, x_vals, xlabel, show_legend=False, legend_loc=''):
    plt.figure(figsize=(3.5, 3))
    plt.show(block=False)
    # Professional color palette
    colors = ['#2E86C1', '#E74C3C', '#27AE60', '#8E44AD', '#F39C12', '#16A085']
    color_idx = 0
    
    for lam, accuracies in all_accuracies.items():
        accuracy_dict = {}
        for filename, accuracy in accuracies:
            basename = os.path.basename(filename)
            pattern_part = os.path.basename(filename).split('_code_ver')[0]
            if pattern_part.endswith('baseline'):
                xname = '1x'
            else:
                xname = format_xname(pattern_part)  # <-- now passing a string
            accuracy_dict[xname] = accuracy
        
        # Replace 'sqrt4x' with '2x' and 'sqrt8x' with '2sqrt2x'
        x_vals = ['2x' if x == 'sqrt4x' else '2sqrt2x' if x == 'sqrt8x' else x for x in x_vals]
        x_vals = ['1/2x' if x == '1/sqrt4x' else '1/2sqrt2x' if x == '1/sqrt8x' else x for x in x_vals]
        x_vals = ['1/4x' if x == '1/sqrt16x' else '4x' if x == 'sqrt16x' else x for x in x_vals]
        accuracy_values = [accuracy_dict.get(x_val, 0) for x_val in x_vals]
        
        if lam == 'avg':
            color = 'black'
            label = 'simple avg of accuracies from each count'
            alpha = 1.0
        else:
            color = colors[color_idx % len(colors)]
            color_idx += 1
            label = f'assuming avg count per area={lam}'
            alpha = 0.7

        plt.plot(x_vals, accuracy_values, label=label, marker='o', 
                color=color, alpha=alpha, linewidth=2, markersize=8)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel(xlabel, fontsize=12)
    # plt.xlabel(r"$\mathrm{}$" "\n" r"$\mathrm{\small{Part\ in\ another\ size}}$", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(-.05, 1.05)
    # plt.title(f'Accuracy vs {xlabel}', fontsize=14, pad=15)
    if show_legend:
        if legend_loc != '':
            plt.legend(fontsize='xx-small', framealpha=0.8, loc=legend_loc)
        else:
            plt.legend(fontsize='xx-small', framealpha=0.8)



    plt.tight_layout()

    # Add horizontal dotted lines
    add_lines = False
    # add_lines = True
    if add_lines:
        y_values = [0.4868, 0.4549, 0.3679]
        for y_val, color in zip(y_values, colors[:len(y_values)]):
            plt.axhline(y=y_val, color=color, linestyle='--', linewidth=1, dashes=(7, 7))
    
    pass

    # Save the plot as an image file
    save_plot = input("Do you want to save the plot as an image file? (y/n): ").strip().lower()
    if save_plot == 'y':
        default_filename = f"accuracy_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"
        filename = input(f"Enter the filename (default: {default_filename}): ").strip()
        if not filename:
            filename = default_filename
        
        plt.savefig(filename, format='svg', )
        print(f"Plot saved to {filename}")

    # Print all data points in the graph
    for lam, accuracies in all_accuracies.items():
        print(f"Lambda: {lam}")
        accuracy_dict = {}
        for filename, accuracy in accuracies:
            basename = os.path.basename(filename)
            pattern_part = os.path.basename(filename).split('_code_ver')[0]
            if pattern_part.endswith('baseline'):
                xname = '1x'
            else:
                xname = format_xname(pattern_part)
            accuracy_dict[xname] = accuracy
        
        # Use the same x_vals processing as the plot
        x_vals_processed = ['2x' if x == 'sqrt4x' else '2sqrt2x' if x == 'sqrt8x' else x for x in x_vals]
        x_vals_processed = ['1/2x' if x == '1/sqrt4x' else '1/2sqrt2x' if x == '1/sqrt8x' else x for x in x_vals_processed]
        x_vals_processed = ['1/4x' if x == '1/sqrt16x' else '4x' if x == 'sqrt16x' else x for x in x_vals_processed]
        
        for x_val in x_vals_processed:
            if x_val in accuracy_dict:
                print(f"{x_val}: {accuracy_dict[x_val]}")

    # Ask in terminal to save the data points as a CSV file
    save_csv = input("Do you want to save the data points as a CSV file? (y/n): ").strip().lower()
    if save_csv == 'y':
        default_filename = f"accuracy_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # default_filename = "accuracy_data.csv"
        filename = input(f"Enter the filename (default: {default_filename}): ").strip()
        if not filename:
            filename = default_filename
        
        # Save the data points to a CSV file
        with open(filename, 'w') as f:
            f.write("Lambda,X Value,Accuracy\n")
            for lam, accuracies in all_accuracies.items():
                accuracy_dict = {}
                for filename_acc, accuracy in accuracies:
                    basename = os.path.basename(filename_acc)
                    pattern_part = basename.split('_code_ver')[0]
                    if pattern_part.endswith('baseline'):
                        xname = '1x'
                    else:
                        xname = format_xname(pattern_part)
                    accuracy_dict[xname] = accuracy
                
                # Use the same x_vals processing as the plot and printing
                x_vals_processed = ['2x' if x == 'sqrt4x' else '2sqrt2x' if x == 'sqrt8x' else x for x in x_vals]
                x_vals_processed = ['1/2x' if x == '1/sqrt4x' else '1/2sqrt2x' if x == '1/sqrt8x' else x for x in x_vals_processed]
                x_vals_processed = ['1/4x' if x == '1/sqrt16x' else '4x' if x == 'sqrt16x' else x for x in x_vals_processed]
                
                for x_val in x_vals_processed:
                    if x_val in accuracy_dict:
                        f.write(f"{lam},{x_val},{accuracy_dict[x_val]}\n")
        print(f"Data points saved to {filename}")


def save_legend_only(all_accuracies, filename=None):
    """
    Create and save just the legend from plot_all_accuracies as an SVG file.
    
    Parameters:
    all_accuracies (dict): Dictionary of accuracies data (same as plot_all_accuracies)
    filename (str): Optional filename for the saved SVG. If None, user will be prompted.
    """
    # Create a figure with transparent background
    fig = plt.figure(figsize=(4, 2))
    fig.patch.set_facecolor('none')  # Transparent background
    
    # Professional color palette (same as plot_all_accuracies)
    colors = ['#2E86C1', '#E74C3C', '#27AE60', '#8E44AD', '#F39C12', '#16A085']
    color_idx = 0
    
    # Create dummy plots just to generate legend entries
    for lam, accuracies in all_accuracies.items():
        if lam == 'avg':
            color = 'black'
            label = 'simple avg of accuracies from each count'
            alpha = 1.0
        else:
            color = colors[color_idx % len(colors)]
            color_idx += 1
            label = f'assuming avg count per area={lam}'
            alpha = 0.7
        
        # Create a dummy line just for the legend
        plt.plot([], [], label=label, marker='o', 
                color=color, alpha=alpha, linewidth=2, markersize=8)
    
    # Create the legend
    plt.show(block=False)
    legend = plt.legend(fontsize='x-small', framealpha=0.8, loc='center')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.5)
    
    # Hide the axes
    # plt.gca().set_visible(False)
    
    # Remove all whitespace around the legend
    plt.tight_layout()
    
    # Get the legend's bounding box
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    
    # Prompt for filename if not provided
    if filename is None:
        default_filename = f"legend_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"
        filename = input(f"Enter the filename for legend (default: {default_filename}): ").strip()
        if not filename:
            filename = default_filename
    
    # Save just the legend area
    plt.savefig(filename, format='svg', bbox_inches=bbox, 
                facecolor='none', edgecolor='none', pad_inches=0.1, )
    print(f"Legend saved to {filename}")
    
    plt.close(fig)  # Close the figure to free memory


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


def generate_x_sequence(min_val, max_val):
    """
    Generates a sequence of x-values in powers of 2 or sqrt(2), 
    depending on whether the user wants sqrt steps or not.
    """

    def format_sqrt(power):
        """
        Interprets 'power' as (sqrt(2))^power = 2^(power/2).
        """
        if power == 0:
            return "1x"
        elif power < 0:
            return f"1/sqrt{2**(-power)}x"
        else:
            return f"sqrt{2**power}x"

    def format_regular(power):
        """
        Interprets 'power' as 2^power.
        """
        if power == 0:
            return "1x"
        elif power < 0:
            return f"1/{2**(-power)}x"
        else:
            return f"{2**power}x"

    # Decide whether to use sqrt(2)-based exponents or 2-based exponents
    use_sqrt = isinstance(min_val, np.float64) or isinstance(max_val, np.float64)

    if use_sqrt:
        # We want (sqrt(2))^p from min_val to max_val.
        # => 2^(p/2) in [min_val, max_val]
        # => p/2 in [log2(min_val), log2(max_val)]
        # => p in [2*log2(min_val), 2*log2(max_val)]
        # The smallest integer p >= 2*log2(min_val) => p_min = ceil(...)
        # The largest integer p <= 2*log2(max_val)  => p_max = floor(...)
        p_min = int(math.ceil(2 * math.log2(min_val)))
        p_max = int(math.floor(2 * math.log2(max_val)))
        steps = range(p_min, p_max + 1)
        return [format_sqrt(p) for p in steps]
    else:
        # We want 2^p from min_val to max_val.
        # => p in [log2(min_val), log2(max_val)]
        p_min = int(math.ceil(math.log2(min_val)))
        p_max = int(math.floor(math.log2(max_val)))
        steps = range(p_min, p_max + 1)
        return [format_regular(p) for p in steps]

# -----------------------
# Usage & test
# -----------------------

# # Folder path containing the CSV files
# folder_path = './processing/scatterstr_test'
folder_path = './processing/scatter'
# folder_path = './processing/background_psf4_peakconst'
# folder_path = './processing/background_test'
# folder_path = './processing/background'
# folder_path = './processing/psfwidth_test_const_peakval_bg'
# folder_path = './processing/snr_test'
# folder_path = './processing/d6-tolerance_of_psf-test'
# folder_path = './processing/zoom_test'
# folder_path = './processing/psfwidth_test_const_peaksum_bg'

# xlabel = "Background Level (psf4, const peak height)"
# xlabel = "Background Level"
xlabel = "Scatter Strength"
# xlabel = "Signal-to-Noise Ratio"
# xlabel = "Analysis PSF width / True PSF width"
# xlabel = "Zoom Factor"
# xlabel = "PSF width (constant: peak value, background)"
# xlabel = "PSF width (constant: peak sum, background)"

# # x_vals = generate_x_sequence(1/np.sqrt(16), np.sqrt(2)# x_vals = generate_x_sequence(1/8, 4)  
# x_vals = generate_x_sequence(1/32, 8)  
# print(x_vals)  
# # Expected: ['1/8x', '1/4x', '1/2x', '1x', '2x', '4x']
# x_vals = generate_x_sequence(1/np.sqrt(4), np.sqrt(8))  
# print(x_vals)  
# # Expected: ['1/sqrt4x', '1/sqrt2x', '1x', 'sqrt2x', 'sqrt4x', 'sqrt8x']
# x_vals = generate_x_sequence(1, np.sqrt(8))  
# print(x_vals)  
# # Expected: ['1x', 'sqrt2x', 'sqrt4x', 'sqrt8x']
# x_vals = generate_x_sequence(1/np.sqrt(8), 1)
# print(x_vals)
# Expected: ['1/sqrt8x', '1/sqrt4x', '1/sqrt2x', '1x'])
# x_vals = generate_x_sequence(1/np.sqrt(8), np.sqrt(8))
# x_vals = generate_x_sequence(1/np.sqrt(8), np.sqrt(16))
# x_vals = generate_x_sequence(1/np.sqrt(16), np.sqrt(2))
# x_vals = generate_x_sequence(1/np.sqrt(16), np.sqrt(8))
# x_vals = generate_x_sequence(1/np.sqrt(2), np.sqrt(8))
# x_vals = generate_x_sequence(1/2, 256)
# x_vals = generate_x_sequence(1/2, 2048)
# x_vals = generate_x_sequence(1/32, 8)
x_vals = generate_x_sequence(1/64, 8)

tags = x_vals

lams = [0.25, 0.5, 1]

# # # # Calculate and print the accuracy for each CSV file
# # avg_accuracies = calculate_avg_accuracies(folder_path)
weighted_accuracies = [calculate_weighted_accuracies(folder_path, lam) for lam in lams]
all_accuracies = {}
# # all_accuracies['avg'] = avg_accuracies
for lam, weighted_accuracy in zip(lams, weighted_accuracies):
    all_accuracies[lam] = weighted_accuracy
plot_all_accuracies(all_accuracies, x_vals, xlabel=xlabel, show_legend=False, legend_loc='lower right') 

# Optionally save just the legend as SVG
save_legend_svg = input("Do you want to save just the legend as an SVG file? (y/n): ").strip().lower()
if save_legend_svg == 'y':
    save_legend_only(all_accuracies)

# # x_vals = generate_x_sequence(1/8, 8)  # Generates: ['1/8x', '1/4x', '1/sqrt(8)x', '1/2x', '1/sqrt(2)x', '1x', 'sqrt(2)x', '2x', 'sqrt(8)x', '4x', '8x']
pass

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

# plot_xi_values(xi_values)

from scipy.special import erf
import numpy as np

def y_function(s, R):
    """
    Computes y = pi * sqrt(s) * erf(R * sqrt(pi * s)).
    
    Parameters:
    - s: array-like or scalar, the input values.
    - R: scalar, the parameter to scale the error function.

    Returns:
    - y: array-like or scalar, the computed y values.
    """
    return np.pi * np.sqrt(s) * erf(R * np.sqrt(np.pi * s))


def plot_prob_of_missing_particle_normalized_by_total_particle_number(psf_widths=[.707, 1.0, 1.41, 2, 2.83, 4.00, 5.66], R_values=[4.1*0.707, 3.1*1, 2.5*1.41, 2.1*2, 1.9*2.83, 1.5*4.00, 0.9*5.66]):
    # Plot: s = 0 to 0.0020 with thicker lines, larger fonts, and tilted x-ticks
    plt.figure(figsize=(12, 14))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(psf_widths)))
    s_6 = np.linspace(0, 0.0020, 100)
    for psf_width, R, color in zip(psf_widths, R_values, colors):
        y_values = y_function(s_6, R)
        plt.plot(s_6, y_values, label=f"PSF width = {psf_width:.2f}, R = {R:.2f}", color=color, linewidth=4)

    plt.xlabel("s", fontsize=24)  # Larger axis label font
    plt.ylabel("y", fontsize=24)  # Larger axis label font
    plt.legend(fontsize=18)  # Larger legend font
    plt.grid(alpha=0.3)

    # Adjust x-ticks and y-ticks for better visibility and tilt x-ticks
    plt.xticks(fontsize=24, rotation=45)  # Tilted x-ticks
    plt.yticks(fontsize=24)  # Larger y-ticks font
    # plt.ylim([0 - 0.005, 0.06])
    plt.tight_layout()
    plt.show(block=False)

    # Prompt user whether to save this png
    save_plot = input("Do you want to save the plot as an image file? (y/n): ").strip().lower()
    if save_plot == 'y':
        default_filename = f"probability_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filename = input(f"Enter the filename (default: {default_filename}): ").strip()
        if not filename:
            filename = default_filename
        
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")

    # Prompt user whether to save the data points as a CSV file
    save_csv = input("Do you want to save the data points as a CSV file? (y/n): ").strip().lower()
    if save_csv == 'y':
        default_filename = f"probability_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filename = input(f"Enter the filename (default: {default_filename}): ").strip()
        if not filename:
            filename = default_filename
        
        # Save the data points to a CSV file
        with open(filename, 'w') as f:
            f.write("s,y,PSF width,R\n")
            for psf_width, R in zip(psf_widths, R_values):
                y_values = y_function(s_6, R)
                for s_val, y_val in zip(s_6, y_values):
                    f.write(f"{s_val},{y_val},{psf_width},{R}\n")
        print(f"Data points saved to {filename}")

    average_slopes = []

    s_range = np.linspace(0, 0.0020, 100)
    for psf_width, R, color in zip(psf_widths, R_values, colors):
        y_values = y_function(s_range, R)
        slopes = np.gradient(y_values, s_range)  # Compute slopes
        avg_slope = np.mean(slopes)  # Compute average slope
        average_slopes.append(avg_slope)

    # Plot average slopes
    plt.figure(figsize=(4, 3))
    plt.bar([f"{psf:.2f}" for psf in psf_widths], average_slopes, color='gray', width=0.6)
    plt.xlabel("PSF Width", fontsize=16)
    plt.ylabel("Average Slope", fontsize=16)
    # plt.title("Average Slope vs. PSF Width", fontsize=18)
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show(block=False)

    # Prompt user whether to save this slope graph
    save_slope_plot = input("Do you want to save the slope plot as an image file? (y/n): ").strip().lower()
    if save_slope_plot == 'y':
        default_slope_filename = f"slope_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        slope_filename = input(f"Enter the filename (default: {default_slope_filename}): ").strip()
        if not slope_filename:
            slope_filename = default_slope_filename
        
        plt.savefig(slope_filename, dpi=300)
        print(f"Slope plot saved to {slope_filename}")

    # Prompt user whether to save the slope data points as a CSV file
    save_slope_csv = input("Do you want to save the slope data points as a CSV file? (y/n): ").strip().lower()
    if save_slope_csv == 'y':
        default_slope_csv_filename = f"slope_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        slope_csv_filename = input(f"Enter the filename (default: {default_slope_csv_filename}): ").strip()
        if not slope_csv_filename:
            slope_csv_filename = default_slope_csv_filename
        
        # Save the slope data points to a CSV file
        with open(slope_csv_filename, 'w') as f:
            f.write("PSF Width,Average Slope\n")
            for psf_width, avg_slope in zip(psf_widths, average_slopes):
                f.write(f"{psf_width},{avg_slope}\n")
        print(f"Slope data points saved to {slope_csv_filename}")

plot_prob_of_missing_particle_normalized_by_total_particle_number()
pass