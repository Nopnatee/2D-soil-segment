import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import torch
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm  # Added missing import

# === FILTER CONFIG ===
BEAD_MASKS = 4             # Number of clusters
CONTRAST_FACTOR = 1.4       # >1 increases contrast
SATURATION_FACTOR = 2.5       # >1 increases vividity
BRIGHTNESS_OFFSET = 0      # Offset for brightness adjustment

# === Load the image path ===
img_path_list = []
folder_path_1 = "pictures/14-7-35"
folder_path_2 = "pictures/15-7-18"
folder_path_3 = "pictures/15-15-15"
folder_path_4 = "pictures/18-4-5"

for path in [folder_path_1, folder_path_2, folder_path_3, folder_path_4]:
    if not os.path.exists(path):
        print(f"Warning: Folder {path} does not exist, skipping...")
        continue
    # Get all image paths from the folders
    image_path = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if image_path:  # Only add if folder contains images
        img_path_list.append(image_path)

if not img_path_list:
    raise FileNotFoundError("No valid image folders found.")

# === Load image to rgb ===
def load_img_as_rgb(img_path):
    image_bgr = cv2.imread(img_path)  # Removed duplicate line
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at: {img_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# === Image enhancement with comprehensive clipping ===
def enhance_image(img_path, contrast=CONTRAST_FACTOR, saturation=SATURATION_FACTOR, brightness=BRIGHTNESS_OFFSET):
    image = load_img_as_rgb(img_path)
    # Create a mask for pure green pixels (RGB = [0, 255, 0])
    green_mask = np.all(image == [0, 255, 0], axis=-1)

    # Backup green pixels
    green_pixels = image[green_mask].copy()  # Added .copy() for safety

    # Ensure input is float32 for precise calculations
    img = np.clip(image.astype(np.float32), 0, 255)

    # Apply contrast with clipping
    img = np.clip(contrast * img, 0, 255)

    # Apply brightness adjustment with clipping
    img = np.clip(img + brightness, 0, 255)

    # Convert to uint8 for HSV conversion
    img_uint8 = img.astype(np.uint8)

    # Convert to HSV and apply saturation boost
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv = np.clip(hsv, 0, 255)

    # Convert back to RGB
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # Restore the original green pixels
    if np.any(green_mask):  # Only restore if there are green pixels
        enhanced[green_mask] = green_pixels

    return enhanced

# === Grouping points ===
def get_all_grouped_points(label, labels, valid_indices, W):
    point_coords = []
    group_indices = np.where(labels == label)[0]
    for indice in group_indices:
        original_flat_index = valid_indices[indice]
        y, x = divmod(original_flat_index, W)
        point_coords.append([x, y])
    return np.array(point_coords)

# === Processing starts here ===
def get_cluster_mask(image_path):
    enhanced_image = enhance_image(image_path)
    H, W, _ = enhanced_image.shape
    pixels = enhanced_image.reshape(-1, 3)

    print("Filtering out pure green pixels...")
    non_green_indices = np.any(pixels != [0, 255, 0], axis=1)  # Fixed variable name
    pixels_non_green = pixels[non_green_indices]  # Fixed variable name

    if len(pixels_non_green) == 0:
        raise ValueError("No non-green pixels found in the image.")

    print("Running KMeans clustering into", BEAD_MASKS, "clusters...")
    kmeans = KMeans(n_clusters=BEAD_MASKS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_non_green)

    print("Preparing valid indices for mapping back...")
    valid_indices = np.where(non_green_indices)[0]  # Fixed variable name

    all_masks = []
    all_point_coords = []

    print("Processing each cluster to create masks and points...")
    # Fixed tqdm usage - removed desc parameter from range()
    for cluster_label in tqdm(range(BEAD_MASKS), desc="Clusters"):
        mask_flat = np.zeros(H * W, dtype=bool)
        cluster_pixel_indices = valid_indices[labels == cluster_label]
        mask_flat[cluster_pixel_indices] = True
        mask_2d = mask_flat.reshape(H, W)
        point_coords = get_all_grouped_points(cluster_label, labels, valid_indices, W)
        all_masks.append(mask_2d)
        all_point_coords.append(point_coords)
        print(f"Cluster {cluster_label}: mask created with {len(point_coords)} points.")

    print("Sorting clusters by total RGB brightness...")

    # Compute average brightness for each mask (proportional, not cumulative)
    brightness_averages = []
    for mask in all_masks:
        masked_pixels = enhanced_image[mask]  # shape: (N, 3) for RGB
        if len(masked_pixels) == 0:
            brightness = 0  # Avoid division by zero
        else:
            brightness = np.mean(masked_pixels)  # Average brightness
        brightness_averages.append(brightness)

    # Get sorted indices from brightest to darkest
    sorted_indices = np.argsort(brightness_averages)[::-1]  # Descending order

    # Reorder clusters by brightness
    all_masks = [all_masks[i] for i in sorted_indices]
    return all_masks

def get_area(image_path):
    all_masks = get_cluster_mask(image_path)
    # === Compute Cluster Areas ===
    cluster_areas = []
    for i, mask in enumerate(tqdm(all_masks, desc="Measuring cluster areas")):
        area = np.sum(mask > 0)  # Counts all non-zero (i.e., masked) pixels
        cluster_areas.append(area)

    # === Print Area Summary ===
    print("\nCluster Area Summary:")
    for i, area in enumerate(cluster_areas):
        print(f"Cluster {i+1}: {area} pixels")

    print("Total Cluster Area:", sum(cluster_areas))
    return cluster_areas

def get_npk(img_path, w_comp, r_comp, s_comp, b_comp, shadow_area):
    cluster_areas = get_area(img_path)
    if len(cluster_areas) < 4:
        raise ValueError("Not enough clusters detected. Expected at least 4 clusters for NPK calculation.")
    
    white_beads = cluster_areas[0]
    stain_beads = cluster_areas[1]
    red_beads = cluster_areas[2]
    black_beads = max(0, cluster_areas[3] - shadow_area)  # Ensure non-negative

    # Fixed: Define compositions properly (removed overwriting of parameters)
    if w_comp is None:
        w_comp = {'N': 46, 'P': 0, 'K': 0}
    if r_comp is None:
        r_comp = {'N': 0, 'P': 0, 'K': 60}
    if s_comp is None:
        s_comp = {'N': 21, 'P': 0, 'K': 0}
    if b_comp is None:
        b_comp = {'N': 18, 'P': 46, 'K': 0}

    npk_total = {'N': 0, 'P': 0, 'K': 0}

    for key in npk_total:
        npk_total[key] += r_comp[key] * int(red_beads)
        npk_total[key] += w_comp[key] * int(white_beads)
        npk_total[key] += s_comp[key] * int(stain_beads)
        npk_total[key] += b_comp[key] * int(black_beads)

    total_beads = max(1, sum(cluster_areas) - shadow_area)  # Avoid division by zero
    npk_composition = {key: round(val / total_beads, 2) for key, val in npk_total.items()}
    
    output_str = (
        f"Approximated N-P-K Composition: {[value for value in npk_composition.values()]}"
    )

    return output_str

# Main execution with error handling
if __name__ == "__main__":
    try:
        if img_path_list and img_path_list[0]:
            result = get_npk(img_path_list[0][0], w_comp=None, r_comp=None, s_comp=None, b_comp=None, shadow_area=0)
            print("NPK Composition:", result)
        else:
            print("No valid images found to process.")
    except Exception as e:
        print(f"Error processing image: {e}")
        print("Please check that:")
        print("1. Image folders exist and contain valid image files")
        print("2. Required libraries are installed (opencv-python, scikit-learn, tqdm)")
        print("3. Images are not corrupted")