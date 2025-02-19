import os
import shutil
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from PIL import Image
import numpy as np
from math import ceil
import argparse
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import json

def divide_images(input_file_path, output_dir, divide_dim, progress_callback=None):
    os.makedirs(output_dir, exist_ok=True)
    with Image.open(input_file_path) as img:
        if img.mode == 'RGB':
            img = img.convert('L')
        img_array = np.array(img)
    img_height, img_width = img_array.shape
    overlap = 0
    num_sub_images_x = ceil((img_width - overlap) / (divide_dim[0] - overlap))
    num_sub_images_y = ceil((img_height - overlap) / (divide_dim[1] - overlap))
    total_sub_images = num_sub_images_x * num_sub_images_y
    processed_sub_images = 0

    for i in range(num_sub_images_y):
        for j in range(num_sub_images_x):
            start_x = j * (divide_dim[0] - overlap)
            start_y = i * (divide_dim[1] - overlap)
            end_x = start_x + divide_dim[0]
            end_y = start_y + divide_dim[1]
            sub_image = img_array[start_y:end_y, start_x:end_x]
            sub_image_filename = f'div_{i:03}_{j:03}.tiff'
            sub_image_path = os.path.join(output_dir, sub_image_filename)
            sub_image_pil = Image.fromarray(sub_image)
            sub_image_pil.save(sub_image_path)
            processed_sub_images += 1
            if progress_callback:
                progress_callback(processed_sub_images, total_sub_images)

def gaussian_2d(x, y, x0, y0, sigma, amplitude, offset):
    return offset + amplitude * np.exp(
        -(((x - x0) ** 2) / (2 * sigma ** 2) + ((y - y0) ** 2) / (2 * sigma ** 2))
    )

def fit_gaussian_2d(image):
    y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    
    # Apply a low-pass filter to the image
    filtered_image = gaussian_filter(image, sigma=3)
    
    # Find the coordinates of the maximum value in the filtered image
    y0, x0 = np.unravel_index(np.argmax(filtered_image), filtered_image.shape)
    
    # Initial guess based on the filtered image
    initial_guess = (x0, y0, 1, np.max(image), np.min(image))
    
    params, _ = curve_fit(gaussian_2d_wrapper, 
                         (y.ravel(), x.ravel()), 
                         image.ravel(), 
                         p0=initial_guess)
    
    return params

def gaussian_2d_wrapper(coordinates, x0, y0, sigma, amplitude, offset):
    y, x = coordinates
    return gaussian_2d(x, y, x0, y0, sigma, amplitude, offset).ravel()

def main():
    parser = argparse.ArgumentParser(description="Divide TIFF images into smaller sub-images and perform Gaussian fitting.")
    parser.add_argument('-f', '--folder', type=str, help="Folder containing TIFF images to be divided")
    parser.add_argument('-s', '--size', type=int, default=100, help="Size of sub-image (sz x sz)")
    parser.add_argument('--terminal', action='store_true', help="Run the process without UI")
    args = parser.parse_args()

    if args.terminal:
        if not args.folder or not args.size:
            print("Error: Folder and size must be provided when running without UI.")
            print("Example: -f /path/to/folder -s 100")
            return
        input_folder = args.folder
        sz = args.size
    else:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Image Division UI", "Welcome to Image Division UI.\n\nTo run this program without UI, use the --terminal flag and provide the folder and size arguments.")
        input_folder = filedialog.askdirectory(title="==== Select Folder Containing all TIFF images to be divided ====")
        if not input_folder:
            messagebox.showerror("Error", "No folder selected. Exiting.")
            return
        while not any(file.endswith('.tiff') for file in os.listdir(input_folder)):
            messagebox.showerror("Error", "No TIFF files found in the selected folder. Please select a folder that contains TIFF files.")
            input_folder = filedialog.askdirectory(title="==== Select Folder Containing all TIFF images to be divided ====")
        if not input_folder:
            print("No folder selected. Exiting.")
            return
        while not os.path.exists(input_folder):
            retry = messagebox.askretrycancel("Error", "Selected folder does not exist. Please select a valid folder or cancel to quit.")
            if not retry:
                print("No valid folder selected. Exiting.")
                return

        sz = simpledialog.askinteger("Input", "Enter sz of sub-image (sz x sz) in pixels:", initialvalue=100, minvalue=1)
        if not sz:
            print("Invalid size. Exiting.")
            return

    if not args.terminal:
        root = tk.Tk()
        root.title("Processing Images")
        progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        progress.pack(pady=20)
        progress_label = tk.Label(root, text="Starting...")
        progress_label.pack()
        current_file_label = tk.Label(root, text="")
        current_file_label.pack()

    def update_progress(processed, total):
        if not args.terminal:
            progress["value"] = (processed / total) * 100
            progress_label.config(text=f"Processed {processed} of {total} sub-images")
            root.update_idletasks()

    short_names = []
    try:
        for file in os.listdir(input_folder):
            if file.endswith('.tiff'):
                input_file_path = os.path.join(input_folder, file)
                n1 = 40 
                n2 = 4
                short_name = file[:n1] + '___' + file.split('.tif')[0][-n2:] if len(file) > 50 else file
                short_names.append(short_name)
                output_dir = os.path.join(input_folder, short_name)
                if not args.terminal:
                    current_file_label.config(text=f"Dividing file: {file}")
                    root.update_idletasks()
                divide_images(input_file_path, output_dir, (sz, sz), update_progress)
                print(f"Processed {file}")
    except FileNotFoundError as e:
        if not args.terminal:
            messagebox.showerror("Error", f"File not found: {e.filename}")
        else:
            print(f"Error: File not found: {e.filename}")
        return

    if not args.terminal:
        root.destroy()
        messagebox.showinfo("Image Division UI", f"Image division completed successfully.\n\nShort names of processed files:\n{', '.join(short_names)}")
    else:
        print("Image division completed successfully.")
        print(f"Short names of processed files:\n{', '.join(short_names)}")

    # Select images for Gaussian fitting
    if args.terminal:
        print("Please choose a folder from the following list to use for determining PSF width:")
        for i, short_name in enumerate(short_names):
            print(f"{i + 1}. {short_name}")
        folder_choice = int(input("Enter the number of the folder you want to use: ")) - 1
        selected_folder = os.path.join(input_folder, short_names[folder_choice])
        
        selected_files = []
        print(f"Please choose at least 3 TIFF files from the folder {selected_folder} that clearly have particles in them:")
        tiff_files = [f for f in os.listdir(selected_folder) if f.endswith('.tiff')]
        for i, tiff_file in enumerate(tiff_files):
            print(f">> {i + 1}. {tiff_file}")
        file_choices = input("Enter the filenames of the files you want to use, separated by commas (you can omit the 'div_' prefix and '.tiff' extension): ").split(',')
        file_choices = [f'div_{choice.strip()}.tiff' if not choice.strip().startswith('div_') else f'{choice.strip()}.tiff' for choice in file_choices]
        for choice in file_choices:
            if choice in tiff_files:
                selected_files.append(os.path.join(selected_folder, choice))
            else:
                print(f"Error: {choice} not found in the folder {selected_folder}.")
        if len(selected_files) < 3:
            print("Error: Please select at least 3 images.")
            return
    else:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Gaussian Fitting", "Please select at least 3 sub(divided)-images that clearly have particles for Gaussian fitting.")
        selected_files = filedialog.askopenfilenames(title="Select Images for Gaussian Fitting", filetypes=[("TIFF files", "*.tiff")])
        if len(selected_files) < 3:
            messagebox.showerror("Error", "Exiting - 3+ images not selected.")
            return

    sigma_values = []
    images = []
    fitted_images = []
    for file in selected_files:
        image = np.array(Image.open(file))
        params = fit_gaussian_2d(image)
        sigma_values.append(params[2])
        fitted_image = gaussian_2d(*np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0])), *params).reshape(image.shape)
        images.append(image)
        fitted_images.append(fitted_image)
        print(f"Fitted parameters for {file}: (x, y) = ({params[0]:0.1f}, {params[1]:0.1f}), sigma = {params[2]:0.1f}, amplitude = {params[3]:0.1f}, background = {params[4]:0.1f}")

    mean_sigma = np.mean(sigma_values)
    if not args.terminal:
        messagebox.showinfo("Gaussian Fitting", f"Gaussian fitting completed. Mean sigma: {mean_sigma:.2f}")

    # Plot the original and fitted images side-by-side for the selected images
    plt.figure(figsize=(5, 11))
    
    # Determine the common color range for the original and fitted images
    vmin = min(np.min(image) for image in images)
    vmax = max(np.max(image) for image in images)
    
    # Plot original and fitted images side-by-side for the selected images
    for i in range(len(images)):
        plt.subplot(len(images), 2, 2 * i + 1)
        plt.title(f'Original {i+1}', fontsize=8)
        plt.imshow(images[i], cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar().ax.tick_params(labelsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.subplot(len(images), 2, 2 * i + 2)
        plt.title(f'Fitting {i+1} - Sigma: {sigma_values[i]:.2f}', fontsize=8)
        plt.imshow(fitted_images[i], cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar().ax.tick_params(labelsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
    
    # Plot histogram of sigma values
    plt.figure(figsize=(4,3))
    plt.hist(sigma_values, bins=20, edgecolor='black')
    plt.axvline(mean_sigma, color='r', linestyle='dashed', linewidth=1)
    plt.text(mean_sigma, plt.ylim()[1] * 0.9, f'Mean: {mean_sigma:.2f}', color='r', fontsize=8)
    plt.title('Histogram of Sigma Values', fontsize=10)
    plt.xlabel('Sigma', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    
    plt.show()

    # Move the folders containing the subdivided images to ./datasets for analysis
    if args.terminal or messagebox.askokcancel("Move Folders", f"The subdivided images folders will now be moved to {os.path.abspath('./datasets')} for analysis"):
        repo_root_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(repo_root_dir, 'datasets')
        os.makedirs(datasets_dir, exist_ok=True)

        for file in os.listdir(input_folder):
            if file.endswith('.tiff'):
                short_name = file[:n1] + '___' + file.split('.tif')[0][-n2:] if len(file) > 50 else file
                output_dir = os.path.join(input_folder, short_name)
                if os.path.isdir(output_dir):
                    new_output_dir = os.path.join(datasets_dir, short_name)
                    if os.path.exists(new_output_dir):
                        shutil.rmtree(new_output_dir)
                    try:
                        os.rename(output_dir, new_output_dir)
                    except OSError:
                        shutil.copytree(output_dir, new_output_dir)
                        shutil.rmtree(output_dir)
                    print(f"Moved {output_dir} to {new_output_dir}")
        print(f"All subdivided images have been moved to {os.path.abspath('./datasets')} for analysis.")
        if not args.terminal:
            moved_folders = [os.path.join(datasets_dir, short_name) for short_name in short_names]
            messagebox.showinfo("Move Folders", f"All subdivided images have been moved to {os.path.abspath('./datasets')} for analysis:\n\n" + "\n".join(moved_folders))

    # Create config JSON files for each dataset
    config_dir = './configs'
    os.makedirs(config_dir, exist_ok=True)
    for short_name in short_names:
        folder_path = os.path.join(datasets_dir, short_name)
        if os.path.isdir(folder_path):
            config_data = {
                "image_folder_namebase": short_name,
                "code_version_date": "2025-02-12",
                "file_format": "tiff",
                "analyze_the_dataset?": True,
                "ana_random_seed": np.random.randint(0, 10000),
                "ana_predefined_psf_sigma": round(mean_sigma, 3),
                "ana_use_premature_hypothesis_choice?": False,
                "ana_maximum_hypothesis_index": 5,
                "ana_delete_the_dataset_after_analysis?": True
            }
            # Present the config data as a list for user to edit
            config_data_list = [
                ("image_folder_namebase", config_data["image_folder_namebase"]),
                ("code_version_date", config_data["code_version_date"]),
                ("file_format", config_data["file_format"]),
                ("analyze_the_dataset?", config_data["analyze_the_dataset?"]),
                ("ana_random_seed", config_data["ana_random_seed"]),
                ("ana_predefined_psf_sigma", config_data["ana_predefined_psf_sigma"]),
                ("ana_use_premature_hypothesis_choice?", config_data["ana_use_premature_hypothesis_choice?"]),
                ("ana_maximum_hypothesis_index", config_data["ana_maximum_hypothesis_index"]),
                ("ana_delete_the_dataset_after_analysis?", config_data["ana_delete_the_dataset_after_analysis?"])
            ]

            config_path = os.path.join(config_dir, f'{short_name}.json')
            with open(config_path, 'w') as config_file:
                json.dump(config_data, config_file, indent=4)
            print(f"Config file created: {os.path.abspath(config_path)}")
            print(json.dumps(config_data, indent=4))
            print(f"The calculated mean_sigma (mean psf width) value {mean_sigma:.3f} is written as the analysis parameter in the config file.")
            print(f"The image_folder_namebase of {short_name} is used.")
            if not args.terminal:
                messagebox.showinfo("Config Files", f"Config file {short_name}.json has been created in {os.path.abspath(config_dir)}:\n\nConfig contents:\n{json.dumps(config_data, indent=4)}\n\nThe calculated mean_sigma (mean psf width) value {mean_sigma:.3f} is written as the analysis parameter in the config file.\nThe image_folder_namebase of {short_name} is used.\n\nPlease open the config file to edit as necessary before running the analysis program.")
        else:
            print(f"Folder {folder_path} does not exist.")
            print(f"Error: Folder {folder_path} does not exist. Config file will not be created.")
            if not args.terminal:
                messagebox.showerror("Error", f"Folder {folder_path} does not exist. Config file will not be created.")

    if not args.terminal:
        if messagebox.askokcancel("Open Config Folder", "Do you want to open the folder containing the newly created config files?"):
            os.startfile(os.path.abspath(config_dir))

    print(f"\n-- Please open the config file {short_name}.json in {os.path.abspath(config_dir)} to edit as necessary before running the analysis program.")
    print("All done.")

if __name__ == "__main__":
    main()