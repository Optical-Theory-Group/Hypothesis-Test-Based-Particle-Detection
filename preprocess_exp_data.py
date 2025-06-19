import os
import shutil
# import sys
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
import cv2
from scipy import ndimage
from skimage import filters, morphology
from datetime import datetime #, timedelta

def rough_count_particles(tiff_path):
    # Step 1: Open and convert to grayscale (average channels)
    with Image.open(tiff_path) as img:
        img = np.array(img)

    # Check if the TIFF file is RGB and convert to grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        # Convert 16-bit RGB to grayscale
        grayscale = np.mean(img, axis=2)
    else:
        # Assume the image is already grayscale
        grayscale = img

    # Step 2: Normalize to [0, 1] based on bit depth
    if img.dtype == np.uint16:  # 16-bit image
        grayscale_norm = grayscale / 65535.0
    elif img.dtype == np.uint8:  # 8-bit image
        grayscale_norm = grayscale / 255.0
    else:
        raise ValueError("Unsupported image bit depth. Only 8-bit and 16-bit images are supported.")

    # crop to middle 50% of the image to avoid dark corners associated with Gaussian beam profile
    height, width = img.shape[:2]
    crop_height = height // 2
    crop_width = width // 2
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2
    img = img[start_y:start_y + crop_height, start_x:start_x + crop_width]

    # Step 3: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(grayscale_norm, (5, 5), 0)

    # Step 4: Threshold (Otsu's method)
    thresh_val = filters.threshold_otsu(blurred)
    binary_mask = blurred > thresh_val

    # Step 5: Remove small objects (likely noise)
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=20)

    # Step 6: Label connected components
    _, num_features = ndimage.label(cleaned_mask)

    return num_features


# tiff_path = next((file for file in os.listdir('./example_large_tiff') if file.endswith('.tiff')), None)
# if tiff_path:
#     tiff_path = os.path.join('./example_large_tiff', tiff_path)
# else:
#     raise FileNotFoundError("No TIFF files found in ./example_large_tiff")
# rough_count = rough_count_particles(tiff_path)


def divide_images(input_file_path, output_dir, divide_dim, overlap=0, progress_callback=None, crop_root_img_fraction=0.7):
    os.makedirs(output_dir, exist_ok=True)
    with Image.open(input_file_path) as img:
        if img.mode == 'RGB':
            img = img.convert('L')
        img_array = np.array(img)
    img_height, img_width = img_array.shape

    # crop down to central image region according to the crop_root_img_fraction to avoid dark corners associated with Gaussian beam profile
    if crop_root_img_fraction > 1 or crop_root_img_fraction < 0:
        print("Warning: Invalid crop fraction value. Using default of 0.7.")
        crop_root_img_fraction = 0.7

    if crop_root_img_fraction < 1:
        crop_pixel_height = int(img_height * crop_root_img_fraction)
        crop_pixel_width = int(img_width * crop_root_img_fraction)
        crop_top_bottom_pixels = (img_height - crop_pixel_height) // 2
        crop_left_right_pixels = (img_width - crop_pixel_width) // 2
        
        img_array = img_array[crop_top_bottom_pixels:crop_top_bottom_pixels + crop_pixel_height,
                              crop_left_right_pixels:crop_left_right_pixels + crop_pixel_width]
        img_height, img_width = img_array.shape

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
            sub_image_filename = f'div_{i:02}_{j:02}.tiff'
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
    
    # Check if there's a meaningful signal in the image
    signal_max = np.max(filtered_image)
    signal_min = np.min(filtered_image)
    signal_mean = np.mean(filtered_image)
    signal_std = np.std(filtered_image)
    
    # Skip images with insufficient contrast (no clear particle)
    if signal_max - signal_min < 3 * signal_std or (signal_max - signal_mean) < 2 * signal_std:
        return None
    
    # Initial guess based on the filtered image
    initial_guess = (x0, y0, 1, np.max(image), np.min(image))
    
    try:
        params, _ = curve_fit(gaussian_2d_wrapper, 
                            (y.ravel(), x.ravel()), 
                            image.ravel(), 
                            p0=initial_guess,
                            maxfev=1000)  # Increase max function evaluations
        
        # Validate results (ensure sigma is positive and reasonable)
        if params[2] <= 0 or params[2] > 100 or params[3] <= 0:
            return None
            
        return params
    except (RuntimeError, ValueError):
        # Catch fitting failures
        return None

def gaussian_2d_wrapper(coordinates, x0, y0, sigma, amplitude, offset):
    y, x = coordinates
    return gaussian_2d(x, y, x0, y0, sigma, amplitude, offset).ravel()

def get_time_differences(filenames):
    """
    Calculates the time differences between consecutive files
    based on HH-MM-SS timestamps in their filenames.

    Args:
        filenames: array of sorted filenames (strings) that contain timestamps

    Returns:
        list: A list of timedelta objects representing the time differences
              between consecutive files, sorted by their timestamps.
              Returns an empty list if there are fewer than two relevant files.
    """
    file_timestamps = []

    # Extract timestamps and associate with full paths
    for filename in filenames:
        if len(filename) >= 9 and filename[2] == '-' and filename[5] == '-' and filename[8] == '_':
            timestamp_str = filename[:8]  # e.g., "HH-MM-SS"
            try:
                # Assuming the files are from the same day, or we only care about time of day differences.
                dt_object = datetime.strptime(timestamp_str, "%H-%M-%S")
                file_timestamps.append((dt_object, filename)) # Store datetime object and original filename
            except ValueError:
                # Handle cases where the format doesn't match despite initial checks
                print(f"Warning: Could not parse timestamp from filename: {filename}")
                continue
        else:
            print(f"Skipping file (does not match expected format): {filename}")

    if not file_timestamps:
        print("No files with matching HH-MM-SS_ format found or parsed successfully.")
        return []


    difference_seconds = []
    # Calculate differences between consecutive files
    for i in range(1, len(file_timestamps)):
        current_time = file_timestamps[i][0]
        previous_time = file_timestamps[i-1][0]

        # Calculate the difference. NB We do not handle cases where the time wraps around midnight.
        difference = current_time - previous_time
        
        # Convert the timedelta object to total seconds
        difference_seconds.append(difference.total_seconds())           

    return difference_seconds

def find_iqr_threshold(data, k=3):
    """
    Calculates the upper bound threshold for outliers using the IQR method.

    The IQR method defines potential outliers as data points that fall
    outside of the range [Q1 - k*IQR, Q3 + k*IQR]. This function specifically
    returns the upper bound (Q3 + k*IQR) as the threshold for 'long' intervals.

    Args:
        data (list or numpy.ndarray): A list or array of numerical data (e.g., time differences).
        k (float): The multiplier for the IQR. Common values are 1.5 (for mild outliers)
                   or 3.0 (for extreme outliers). Defaults to 1.5.

    Returns:
        float: The calculated upper bound threshold. Returns 0 if data is empty or too short.
    """
    if not data or len(data) < 2:
        print("Warning: Data is empty or has too few elements to calculate quartiles. Returning 0.")
        return 0.0

    # Convert to numpy array for robust percentile calculations
    data_np = np.array(data)

    # Calculate Q1 (25th and 75th percentile)
    Q1 = np.percentile(data_np, 25)
    Q3 = np.percentile(data_np, 75)

    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Calculate the upper bound threshold
    upper_bound_threshold = Q3 + k * IQR

    return upper_bound_threshold


def get_intervaled_tiffs_files(all_tiff_files, interval=0):
    # we use the time stamps in the file name to calculate interval between each image and the select the first image occuring after a large inteveral. 
    # The large interval is found automatically using an interquartile range method. 
    # These images should then correspond to images from distinct image runs in different sample regions.
    
    if interval == 0:
        time_diffs = get_time_differences(sorted(all_tiff_files))
        threshold = find_iqr_threshold(time_diffs)
        print(f"Threshold image interval is {threshold} seconds.")

        big_interval_inds = np.where(time_diffs >= threshold)[0] + 1
        img_inds = np.insert(big_interval_inds, 0, 0)

        intervaled_tiff_files = [all_tiff_files[i] for i in img_inds]
    else:
        intervaled_tiff_files = all_tiff_files[::interval]

    return intervaled_tiff_files


def main():
    parser = argparse.ArgumentParser(description="Divide TIFF images into smaller sub-images and perform Gaussian fitting.")
    parser.add_argument('-t', '--terminal', action='store_true', help="Run the process without UI")
    parser.add_argument('-f', '--folder', type=str, help="Folder containing TIFF images to be divided")
    parser.add_argument('-s', '--size', type=int, help="Size of sub-image (sz x sz)")
    parser.add_argument('-o', '--overlap', type=int, default=0, help="Overlap size in pixels")
    parser.add_argument('-i', '--interval', type=int, default=0, help="Process every Nth file (e.g., 2 means every 2nd file). 0 implies an IQR method timestamps method is used.")
    parser.add_argument('-c', '--crop', type=int, default=0.7, help="Crop down size of raw image files (e.g., 0.7 means image height and width will be reduced to 0.7 of the original size)")
    parser.add_argument('--save-plots', action='store_true', help="Save plots to file instead of displaying (useful for headless environments)")
    args = parser.parse_args()

    
  
    if args.terminal:
        if not args.folder:
            print("Error: Folder must be provided.")
            return
        else:
            # Clean up the path - remove quotes and normalize path
            # Remove trailing slashes or backslashes
            input_folder = args.folder
            input_folder = input_folder.rstrip('\\/')
            input_folder = args.folder.strip('\'"')
            # Make sure path exists
            if not os.path.isdir(input_folder):
                print(f"Error: Folder '{input_folder}' does not exist or is not accessible.")
                return
                
        # Process file interval setting
        interval = args.interval
        if interval < 0:
            print("Warning: Invalid interval value. Using default of 1 (process all files).")
            interval = 1
        elif interval == 0:
            print("Processing files based on timestamp intervals.")
        elif interval > 1:
            print(f"Processing every {interval}th file.")

        # List TIFF files in the folder
        try:
            all_tiff_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.tiff')])
            if not all_tiff_files:
                print(f"Warning: No TIFF files found in '{input_folder}'")
                all_tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tiff', '.tif'))]
                if all_tiff_files:
                    print(f"Found {len(all_tiff_files)} files with .tif extension instead")
                else:
                    print(f"No TIFF or TIF files found in the folder. Exiting.")
                    return

            intervaled_tiff_files = get_intervaled_tiffs_files(all_tiff_files, interval)
            print(f"Intervaled (interval: {interval}) TIFF files to be divided into subimages:")
            for i, file in enumerate(intervaled_tiff_files):
                print(f"{i + 1}. {file}")
        except Exception as e:
            print(f"Error accessing folder or listing files: {e}")
            return

        if not args.size: 
            print("Sub-image size not provided.")

            # Ask user if they want to automatically calculate sub-image size
            print("\nDo you want to automatically calculate sub-image size?")
            print("1. Yes")
            print("2. No, I will provide the size")
            choice = input("Enter your choice (1 or 2): ")

            if choice == '1':
                # List TIFF files in the folder
                print("\nIntervaled TIFF files to be divided into subimages:")
                for i, file in enumerate(intervaled_tiff_files):
                    print(f"{i + 1}. {file}")

                # Get user selection
                choice = int(input("\nSelect a file number to estimate sub-image size: ")) - 1
                if 0 <= choice < len(intervaled_tiff_files):
                    chosen_tiff_path = os.path.join(input_folder, intervaled_tiff_files[choice])
                    particle_count = rough_count_particles(chosen_tiff_path)
                    # Suggest a size that would result in average of 1 particles per sub-image (P(n=5) ~ 0.003 in this case thus testing up to n=4 covers 99.7% of the cases)
                    with Image.open(chosen_tiff_path) as img:
                        width, height = img.size
                    total_area = width * height
                    area_per_particle = total_area / particle_count
                    sz = int(np.sqrt(area_per_particle * 1))
                    print(f"\n(Rougly) estimated {particle_count} particles in the entire image.")
                    print(f"Suggested sub-image: {sz}x{sz} pixels")
                    if not 20 < sz < 200:
                        print("Warning: The suggested size is outside the range of 20 to 200 pixels. Clipping to this range.")
                        print("If you want to use size outside this range, please run the program again with the '--size' (or '-s') argument.") 
                        if sz < 20:
                            sz = 20
                        elif sz > 200:
                            sz = 200
                else:
                    print("Invalid selection, using default size of 100")
                    return
            elif choice == '2':
                sz = int(input("Enter the desired sz of sub-image (sz x sz) in pixels: "))
            else:
                print("Invalid choice. Please enter 1 or 2.")
                return
        else:
            sz = args.size


        if not args.overlap:
            print("Overlap size not provided. Setting to default (0 pixels).")
            overlap = 0

    else:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Image Division UI", "Welcome to Image Division UI.\n======================\nTo in terminal instead, use the --terminal flag with the -f (folder, required) argument (-s (size), -o (overlap), and -i (interval) are optional).")
        input_folder = filedialog.askdirectory(title="==== Select Folder Containing all TIFF images to be divided ====")
        if not input_folder:
            messagebox.showerror("Error", "No folder selected. Exiting.")
            return
        while not any(file.endswith('.tiff') for file in os.listdir(input_folder)):
            messagebox.showerror("Error", "No TIFF files found in the selected folder.")
            input_folder = filedialog.askdirectory(title="==== Select Folder Containing all TIFF images to be divided ====")
        if not input_folder:
            print("No folder selected. Exiting.")
            return
        while not os.path.exists(input_folder):
            retry = messagebox.askretrycancel("Error", "Selected folder does not exist.")
            if not retry:
                print("No valid folder selected. Exiting.")
                return

        interval = simpledialog.askinteger("Tiff file skip interval", "Enter the interval for processing TIFF files (e.g., 2 means every 2nd file):", initialvalue=1, minvalue=1)

        all_tiff_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.tiff')])

        intervaled_tiff_files = get_intervaled_tiffs_files(all_tiff_files) 

        files_list = "\n".join(intervaled_tiff_files)

        # Ask user if they want to automatically calculate sub-image size
        auto_size = messagebox.askyesno("Sub-image Size", f"The following files will be processed:\n{files_list}\n================\n\nSuggest an appropriate sub-image size based on a rough estimate of particle density?")

        if auto_size:
            # List TIFF files in the folder
            # Create dialog to select a file
            dialog = tk.Toplevel()
            dialog.title("Select TIFF File")
            dialog.geometry("700x300")  # Set width=400, height=300
            listbox = tk.Listbox(dialog, width=700)
            for file in intervaled_tiff_files:
                listbox.insert(tk.END, file)
            listbox.pack(padx=10, pady=10)

            # Enable double-click selection
            def on_double_click(event):
                selected_file[0] = listbox.get(listbox.curselection())
                dialog.destroy()
            
            listbox.bind('<Double-Button-1>', on_double_click)
            
            selected_file = [None]
            def on_select():
                selected_file[0] = listbox.get(listbox.curselection())
                dialog.destroy()
            
            tk.Button(dialog, text="Select", command=on_select).pack(pady=5)
            dialog.wait_window()
            
            if selected_file[0]:
                tiff_path = os.path.join(input_folder, selected_file[0])
                particle_count = rough_count_particles(tiff_path)
                with Image.open(tiff_path) as img:
                    width, height = img.size
                total_area = width * height
                area_per_particle = total_area / particle_count
                sz = int(np.sqrt(area_per_particle * 1))
                if sz < 20:
                    clip_yesno = messagebox.askyesno("Warning", f"Suggested {sz} pixels is below reasonable range (20-200). Set it to 20?")
                    if clip_yesno:
                        sz = 20
                    else:
                        sz = simpledialog.askinteger("Confirm Size", "Enter desired sub-image size:", initialvalue=sz, minvalue=10)
                elif sz > 200:
                    clip_yesno = messagebox.askyesno("Warning", f"Suggested {sz} pixels is above reasonable range (20-200). Set it to 200?")
                    if clip_yesno:
                        sz = 200
                    else:
                        sz = simpledialog.askinteger("Confirm Size", "Enter desired sub-image size:", initialvalue=sz, minvalue=10)
                else:
                    sz = simpledialog.askinteger("Confirm Size", f"Estimated {particle_count} particles in the entire image. \nSuggested sub-image size: {sz}x{sz}. If you want to use a different size, enter differently.", initialvalue=sz, minvalue=10)
                    messagebox.showinfo("Confirm Size", f"Estimated {particle_count} particles in the entire image. \nSuggested sub-image size: {sz}x{sz}")

            else:
                messagebox.showerror("Error", "No file selected.")
                sz = simpledialog.askinteger("Input", "Enter the desired sz of sub-image (sz x sz) in pixels:", initialvalue=100, minvalue=10)
        else:
            sz = simpledialog.askinteger("Input", "Enter the desired sz of sub-image (sz x sz) in pixels:", initialvalue=100, minvalue=1)
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
        else:
            # Add terminal progress indicator
            percent = int((processed / total) * 100)
            bar_length = 30
            filled_length = int(bar_length * processed // total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r[{bar}] {percent}% - {processed}/{total} sub-images processed', end='', flush=True)
            if processed == total:
                print()  # Add newline after completion

    short_names = []
    if not args.terminal:
        overlap = simpledialog.askinteger("Input", "Enter desired overlap in pixels (default 0):", initialvalue=0, minvalue=0)
        if overlap is None:
            messagebox.showerror("Error", "No folder selected. Exiting.")
            return
        files_list = "\n".join([f"- {file}" for file in intervaled_tiff_files])
        messagebox.showinfo("Starting image division", f"Division of {len(intervaled_tiff_files)} tiff files into {sz}x{sz} subimages with {overlap} overlap will now begin (unit: pixel).\n\n{files_list}")
    else:
        overlap = args.overlap
        print(f"Division into {sz}x{sz} subimages with {overlap} overlap will now begin for intervaled tiff files (unit: pixel).")
        
    # Perform image division
    crop_img_fraction = args.crop
    try:
        if not intervaled_tiff_files:
            raise FileNotFoundError("No TIFF files found in the selected folder.")
        
        print(f"Processing {len(intervaled_tiff_files)} of {len(all_tiff_files)} TIFF files (interval={interval}).")
        
        for file in intervaled_tiff_files:
            input_file_path = os.path.join(input_folder, file)
            n1 = 40 
            n2 = 4
            short_name = file[:n1] + '___' + file.split('.tif')[0][-n2:] if len(file) > 50 else file
            # short_name = file.split('.tif')[0]
            short_names.append(short_name)
            output_dir = os.path.join(input_folder, short_name)
            if not args.terminal:
                current_file_label.config(text=f"Dividing file: {file}")
                root.update_idletasks()
            divide_images(input_file_path, output_dir, (sz, sz), overlap, update_progress, crop_img_fraction)
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
            output_dir = os.path.join(input_folder, short_name)
            num_files = len(os.listdir(output_dir))
            print(f">> {i + 1}. {short_name} ({num_files} files)")
        folder_choice = int(input("Enter the number of the folder you want to use: ")) - 1
        selected_folder = os.path.join(input_folder, short_names[folder_choice])
        
        selected_files = []
        print(f"Please choose at least 3 TIFF files from the folder {selected_folder} that clearly have particles in them:")
        tiff_files = [f for f in os.listdir(selected_folder) if f.endswith('.tiff')]
        
        # Display files with numbers (show only the first 20 if there are many)
        max_display = 20
        if len(tiff_files) > max_display:
            for i in range(max_display):
                print(f">> {i + 1}. {tiff_files[i]}")
            print(f"... and {len(tiff_files) - max_display} more files")
        else:
            for i, tiff_file in enumerate(tiff_files):
                print(f">> {i + 1}. {tiff_file}")
            
        # Ask user to select files by numbers
        # Default to the first 10% of files if the user presses Enter
        default_count = max(1, len(tiff_files) // 10)  # Ensure at least 1 file is selected
        default_selections = ', '.join(str(i + 1) for i in range(min(default_count, len(tiff_files))))
        
        prompt = f"Enter file numbers (e.g., '1, 4, 6, 11' or '1-5' or '1, 2, 4-7, 9') or press Enter for default (first {min(default_count, len(tiff_files))} files): "
        selections = input(prompt).strip()
        
        if not selections:
            selections = default_selections
            print(f"Using default selections: {selections}")
        
        # Parse the selections
        selected_indices = []
        for part in selections.split(','):
            part = part.strip()
            if '-' in part:
                # Handle ranges (e.g., "1-5")
                start, end = map(int, part.split('-'))
                selected_indices.extend(range(start, end + 1))
            else:
                # Handle individual numbers
                try:
                    selected_indices.append(int(part))
                except ValueError:
                    print(f"Warning: Ignoring invalid input '{part}'")
                
        # Convert to 0-based indices and get file paths
        selected_files = []
        for idx in selected_indices:
            if 1 <= idx <= len(tiff_files):
                selected_files.append(os.path.join(selected_folder, tiff_files[idx - 1]))
            else:
                print(f"Warning: Ignoring out-of-range index {idx}")
            
        if len(selected_files) < 3:
            print("Error: Please select at least 3 images.")
            return
    else:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Gaussian Fitting", "Please select at least 3 sub(divided)-images that clearly have particles. The images will be used to determine the PSF width. \nNote: only one particle (likely the brightest) per image will be analyzed.")
        # Get the last folder with divided images
        last_divided_folder = None
        for short_name in short_names:
            folder_path = os.path.join(input_folder, short_name)
            if os.path.isdir(folder_path):
                last_divided_folder = folder_path

        selected_files = filedialog.askopenfilenames(
            title="Select Images for Gaussian Fitting",
            filetypes=[("TIFF files", "*.tiff")],
            initialdir=last_divided_folder
        )
        if len(selected_files) < 3:
            messagebox.showerror("Error", "Exiting - 3+ images not selected.")
            return

    sigma_values = []
    images = []
    fitted_images = []
    skipped_files = []

    for file in selected_files:
        image = np.array(Image.open(file))
        params = fit_gaussian_2d(image)
        
        if params is None:
            print(f"Skipping {os.path.basename(file)} - no clear particle detected")
            skipped_files.append(os.path.basename(file))
            continue
            
        sigma_values.append(params[2])
        fitted_image = gaussian_2d(*np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0])), *params).reshape(image.shape)
        images.append(image)
        fitted_images.append(fitted_image)
        print(f"Fitted parameters for {file}: (x, y) = ({params[0]:0.1f}, {params[1]:0.1f}), sigma = {params[2]:0.1f}, amplitude = {params[3]:0.1f}, background = {params[4]:0.1f}")

    # Calculate mode from histogram
    hist, bin_edges = np.histogram(sigma_values, bins=20)
    max_freq = np.max(hist)
    mode_bin_indices = np.where(hist == max_freq)[0]
    
    # Only proceed with mode filtering if there's a single clear mode
    if len(mode_bin_indices) == 1:
        mode = (bin_edges[mode_bin_indices[0]] + bin_edges[mode_bin_indices[0] + 1]) / 2
        
        # Create a filter mask for values within 0.25*mode to 2*mode
        filter_mask = [(0.25 * mode <= s <= 2 * mode) for s in sigma_values]
        
        # Apply filter to all three arrays simultaneously to keep them in sync
        filtered_sigma_values = [s for i, s in enumerate(sigma_values) if filter_mask[i]]
        filtered_images = [img for i, img in enumerate(images) if filter_mask[i]]
        filtered_fitted_images = [img for i, img in enumerate(fitted_images) if filter_mask[i]]
        
        # Count how many were filtered out
        filtered_out_count = len(sigma_values) - len(filtered_sigma_values)
        
        # Replace the original arrays with the filtered versions
        sigma_values = filtered_sigma_values
        images = filtered_images
        fitted_images = filtered_fitted_images
        
        print(f"\nMode sigma: {mode:.2f}")
        print(f"Filtered {filtered_out_count} outlier values")
    else:
        print("\nMultiple modes detected - skipping outlier filtering")

    if not sigma_values:
        if not args.terminal:
            messagebox.showerror("Error", "No valid particles detected in any of the selected images. Please select different images.")
        else:
            print("ERROR: No valid particles detected in any of the selected images. Please select different images.")
        return

    mean_sigma = np.mean(sigma_values)
    if not args.terminal:
        messagebox.showinfo("Gaussian Fitting", f"Gaussian fitting completed. Mean sigma: {mean_sigma:.2f}")
    else:
        print(f"Mean sigma: {mean_sigma:.2f}")

    # Plot the original and fitted images side-by-side for the selected images
    plt.figure(figsize=(5, 9))
    
    # Determine the common color range for the original and fitted images
    vmin = min(np.min(image) for image in images)
    vmax = max(np.max(image) for image in images)
    
    # Plot original and fitted images side-by-side for the selected images
    n_plot = min(5, len(images))  # Ensure at least 5 plots are shown
    for i in range(n_plot):
        plt.subplot(n_plot, 2, 2 * i + 1)
        plt.title(f'Original {i+1}', fontsize=8)
        plt.imshow(images[i], cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar().ax.tick_params(labelsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.subplot(n_plot, 2, 2 * i + 2)
        plt.title(f'Fitting {i+1} - Sigma: {sigma_values[i]:.2f}', fontsize=8)
        plt.imshow(fitted_images[i], cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar().ax.tick_params(labelsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
    
    # Save or show the fitting plots
    if args.terminal and args.save_plots:
        fits_plot_path = os.path.join(input_folder, "gaussian_fits.png")
        plt.savefig(fits_plot_path)
        print(f"Gaussian fits plot saved to {fits_plot_path}")
    
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
    
    # Save or show the histogram plot
    if args.terminal and args.save_plots:
        hist_plot_path = os.path.join(input_folder, "sigma_histogram.png")
        plt.savefig(hist_plot_path)
        print(f"Sigma histogram plot saved to {hist_plot_path}")
    else:
        plt.show(block=False)

    # Move the folders containing the subdivided images to ./datasets for analysis
    if args.terminal or messagebox.askokcancel("Move Folders", f"The subdivided images folders will now be moved to {os.path.abspath('./datasets')} for analysis"):
        repo_root_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(repo_root_dir, 'datasets')
        os.makedirs(datasets_dir, exist_ok=True)

        for file in os.listdir(input_folder):
            if file.endswith('.tiff'):
                short_name = file[:n1] + '___' + file.split('.tif')[0][-n2:] if len(file) > 50 else file
                # short_name = file.split('.tif')[0]
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
    skipall_edits = False
    for short_name in short_names:
        folder_path = os.path.join(datasets_dir, short_name)
        if os.path.isdir(folder_path):
            config_data = {
                "image_folder_namebase": short_name,
                "code_version_date": "2025-05-02",
                "file_format": "tiff",
                "analyze_the_dataset?": True,
                "ana_random_seed": np.random.randint(0, 10000),
                "ana_predefined_psf_sigma": round(mean_sigma, 3),
                "ana_use_premature_hypothesis_choice?": False,
                "ana_maximum_hypothesis_index": 5,
                "ana_delete_the_dataset_after_analysis?": False
            }

            # Allow terminal users to edit config before saving
            if args.terminal and not skipall_edits:
                print("\n==== Configuration for", short_name, "====")
                print(json.dumps(config_data, indent=4))
                edit_config = input("\nEdit this configuration before saving? Yes (y), No (n) or No for all remaining (na): ").strip().lower()
                if edit_config == 'y':
                    print("\nYou can edit the following parameters:")
                    for i, (key, value) in enumerate(config_data.items(), 1):
                        print(f"{i}. {key}: {value}")
                    
                    while True:
                        param_num = input("\nEnter parameter number to edit (or 'done' to finish): ")
                        if param_num.lower() == 'done':
                            break
                        try:
                            param_num = int(param_num)
                            if 1 <= param_num <= len(config_data):
                                key = list(config_data.keys())[param_num - 1]
                                current_value = config_data[key]
                                print(f"Current value of '{key}': {current_value}")
                                
                                # Handle different types of values
                                if isinstance(current_value, bool):
                                    new_value = input(f"Enter new value (true/false): ").strip().lower()
                                    config_data[key] = new_value == 'true'
                                elif isinstance(current_value, int):
                                    new_value = input(f"Enter new value (integer): ").strip()
                                    config_data[key] = int(new_value)
                                elif isinstance(current_value, float):
                                    new_value = input(f"Enter new value (float): ").strip()
                                    config_data[key] = float(new_value)
                                else:
                                    new_value = input(f"Enter new value (string): ").strip()
                                    config_data[key] = new_value
                                
                                print(f"Updated '{key}' to: {config_data[key]}")
                            else:
                                print("Invalid parameter number!")
                        except ValueError:
                            print("Please enter a valid number or 'done'")
                    
                    print("\nFinal configuration:")
                    print(json.dumps(config_data, indent=4))
                elif edit_config == 'na':
                    skipall_edits = True
            
            config_path = os.path.join(config_dir, f'{short_name}.json')
            with open(config_path, 'w') as config_file:
                json.dump(config_data, config_file, indent=4)
            
            if args.terminal:
                print(f"\nConfig file saved to: {os.path.abspath(config_path)}")
                print("=" * 50)
            else:
                config_path = os.path.join(os.path.abspath(config_dir), f"{short_name}.json")
                message = (
                    f"A new config file has been created - {config_path}:\n\n"
                    f"Config contents:\n{json.dumps(config_data, indent=4)}\n\n"
                    f"The calculated mean_sigma (mean psf width) value {mean_sigma:.3f} "
                    "is written as the analysis parameter in the config file.\n"
                    f"The image_folder_namebase of {short_name} is used.\n\n"
                    "Please open the config file to edit as necessary before running the analysis program."
                )
                messagebox.showinfo("Config Files", message)
        else:
            print(f"Folder {folder_path} does not exist.")
            print(f"Error: Folder {folder_path} does not exist. Config file will not be created.")
            if not args.terminal:
                messagebox.showerror("Error", f"Folder {folder_path} does not exist. Config file will not be created.")
                return

    if not args.terminal:
        if messagebox.askokcancel("Open Config Folder", "Do you want to open the folder containing the newly created Config Files?"):
            os.startfile(os.path.abspath(config_dir))

    if args.terminal:
        print("\n" + "=" * 70)
        print(f"NEXT STEPS:")
        print(f"1. Review and modify the config files in: {os.path.abspath(config_dir)}")
        print(f"2. Run the analysis program with these config files")
        print("=" * 70)

    print(f"\n-- Please open the config file {short_name}.json in {os.path.abspath(config_dir)} to edit as necessary before running the analysis program.")
    print("All done.")

if __name__ == "__main__":
    # if 'pydevd' in sys.modules or 'debugpy' in sys.modules:
    # #     # Run the main function without parallel processing ('-p' option value is False)
    #     sys.argv = ["preprocess_exp_data.py", "--folder", "./datasets/coreshell_np_data/130125_Au_nanoshell_Ag/130125_covidvirus_ctrl", "--terminal", "--size", "50", "--interval", "0", "--save-plot"] 
    #     # ['main.py', '-c', './configs/'] # -p for profiling. Default is False, and it will run on multiple processes.

  # args = 

    main()