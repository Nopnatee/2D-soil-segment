import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
# Ensure the notebook can import the custom U-Net module
cwd = os.getcwd()
current_file = Path(__file__).resolve()
src_candidate = current_file.parents[1] if current_file.parents[1].name == "src" else None

if src_candidate is None or not src_candidate.exists():
    src_candidate = None
    for parent in current_file.parents:
        potential = parent / "src"
        if potential.exists():
            src_candidate = potential
            break

if src_candidate is None:
    raise FileNotFoundError("Could not locate the 'src' directory needed to import custom_unet.")

if str(src_candidate) not in sys.path:
    sys.path.insert(0, str(src_candidate))

PROJECT_ROOT = src_candidate.parent
DEFAULT_REGRESSOR_DATASET_DIR = str(PROJECT_ROOT / "datasets" / "regressor_dataset")

from predictions.npk_pixel_predictor import (
    CLASS_NAME_TO_RAW_MATERIAL,
    DEFAULT_PELLET_CLASS_NAMES,
    RAW_MATERIAL_NUTRIENTS,
)

# Resolve checkpoints directory relative to current working directory
def resolve_checkpoint_path(filename="best_model.pth"):
    search_dir = cwd
    while True:
        candidate = os.path.join(search_dir, "checkpoints", filename)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent
    raise FileNotFoundError(f"Checkpoint '{filename}' not found relative to {cwd}")


# Import your custom U-Net
from soil_segment.custom_unet import SimpleUNet, ConvBlock

# === FILTER CONFIG ===
CONTRAST_FACTOR = 1        # >1 increases contrast
SATURATION_FACTOR = 1      # >1 increases vividity
BRIGHTNESS_OFFSET = 0      # Offset for brightness adjustment
NUM_CLASSES = None         # Will be inferred from the checkpoint
CUDA_WARNING_ISSUED = False
CLASS_NAMES = []
PELLET_CLASS_NAMES = []

# Check GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# === Load U-Net Model ===
def resolve_class_names(num_classes):
    defaults = list(DEFAULT_PELLET_CLASS_NAMES)
    if len(defaults) < num_classes:
        for idx in range(len(defaults), num_classes):
            defaults.append(f"Pellet Class {idx}")
    return defaults[:num_classes]


def _normalize_class_name(name):
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _material_key_for_class(name):
    normalized = _normalize_class_name(name)
    material_key = CLASS_NAME_TO_RAW_MATERIAL.get(normalized)
    if material_key:
        return material_key
    for candidate in RAW_MATERIAL_NUTRIENTS:
        if candidate in normalized:
            return candidate
    return None


def load_unet_model():
    """Load the trained U-Net model and infer its number of classes."""
    checkpoint_path = resolve_checkpoint_path("best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if 'final_conv.weight' not in state_dict:
        raise KeyError("Checkpoint is missing 'final_conv.weight'; cannot infer number of classes.")

    num_classes = state_dict['final_conv.weight'].shape[0]
    model = SimpleUNet(in_channels=3, n_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    global NUM_CLASSES, CLASS_NAMES, PELLET_CLASS_NAMES
    NUM_CLASSES = num_classes
    CLASS_NAMES = resolve_class_names(NUM_CLASSES)
    PELLET_CLASS_NAMES = CLASS_NAMES[1:] if len(CLASS_NAMES) > 1 else []
    print(f"Loaded U-Net with {NUM_CLASSES} classes.")
    return model

# Initialize model
unet_model = load_unet_model()

# U-Net preprocessing transform
unet_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # must match training size
    transforms.ToTensor(),          # Converts to tensor [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # same normalization as training
])

# === Auto-detect image folders ===
def auto_detect_image_folders(base_path=DEFAULT_REGRESSOR_DATASET_DIR):
    """
    Automatically detect all subfolders in the base path that contain images
    
    Args:
        base_path (str): Base directory to search for image folders
    
    Returns:
        list: List of lists containing image paths for each folder
    """
    img_path_list = []
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Warning: Base path '{base_path}' does not exist.")
        return img_path_list
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(base_path) 
               if os.path.isdir(os.path.join(base_path, d))]
    
    # Sort subdirectories for consistent ordering
    subdirs.sort()
    
    print(f"Found {len(subdirs)} subdirectories in '{base_path}':")
    
    # Process each subdirectory
    for subdir in subdirs:
        folder_path = os.path.join(base_path, subdir)
        
        # Get all image files in this folder
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(image_extensions)]
        
        if image_files:
            # Create full paths for images
            image_paths = [os.path.join(folder_path, f) for f in image_files]
            # Sort image paths for consistent ordering
            image_paths.sort()
            img_path_list.append(image_paths)
            print(f"  ✓ {subdir}: {len(image_files)} images")
        else:
            print(f"  ✗ {subdir}: No images found")
    
    if not img_path_list:
        print("No valid image folders found.")
    else:
        print(f"\nTotal folders with images: {len(img_path_list)}")
        total_images = sum(len(folder_images) for folder_images in img_path_list)
        print(f"Total images found: {total_images}")
    
    return img_path_list

# === Load the image paths automatically ===
img_path_list = auto_detect_image_folders()

if not img_path_list:
    raise FileNotFoundError("No valid image folders found.")

# === Load image to rgb ===
def load_img_as_rgb(img_path):
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at: {img_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# === U-Net prediction function ===
def predict_with_unet(image_path, model):
    """
    Use U-Net to predict segmentation masks instead of K-means clustering
    
    Args:
        image_path (str): Path to the input image
        model: Trained U-Net model
    
    Returns:
        numpy.ndarray: Predicted segmentation mask
        numpy.ndarray: Enhanced image for visualization
    """
    # Load and enhance the image
    rgb_image = load_img_as_rgb(image_path)
    
    # Convert enhanced image to PIL for U-Net preprocessing
    enhanced_pil = Image.fromarray(rgb_image.astype(np.uint8))
    
    # Preprocess for U-Net
    input_tensor = unet_transform(enhanced_pil).unsqueeze(0).to(DEVICE)
    
    # Get U-Net prediction
    with torch.no_grad():
        output = model(input_tensor)  # Shape: [1, num_classes, H, W]
    
    # Get predicted mask
    if output.shape[1] == 1:
        # Binary segmentation
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
    else:
        # Multi-class segmentation
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Resize prediction back to original image size
    original_size = rgb_image.shape[:2]
    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), 
                                  (original_size[1], original_size[0]), 
                                  interpolation=cv2.INTER_NEAREST)
    
    return pred_mask_resized, rgb_image

# === GPU-accelerated clustering replacement with U-Net ===
def get_cluster_mask_gpu(image_path, use_gpu=True):
    """
    Replace K-means clustering with U-Net prediction.
    Returns masks following the trained class order (background excluded).
    """
    global CUDA_WARNING_ISSUED
    if use_gpu and not torch.cuda.is_available():
        if not CUDA_WARNING_ISSUED:
            print("Warning: CUDA not available; falling back to CPU inference.")
            CUDA_WARNING_ISSUED = True
        use_gpu = False
    if not PELLET_CLASS_NAMES:
        raise RuntimeError("Class names not initialized; load_unet_model() must be called first.")
    
    # Get U-Net prediction
    pred_mask, enhanced_image = predict_with_unet(image_path, unet_model)
    
    # Create individual masks for each class (excluding background class 0)
    pellet_masks = []
    for class_id, _ in enumerate(PELLET_CLASS_NAMES, start=1):
        class_mask = pred_mask == class_id
        pellet_masks.append(class_mask)

    return pellet_masks, enhanced_image

# === GPU-accelerated area computation ===
def get_area_gpu(image_path, use_gpu=True):
    all_masks, enhanced_image = get_cluster_mask_gpu(image_path, use_gpu)
    
    # Compute cluster areas
    cluster_areas = []
    if use_gpu and torch.cuda.is_available():
        for i, mask in enumerate(tqdm(all_masks, desc="Measuring cluster areas")):
            mask_tensor = torch.from_numpy(mask).to(DEVICE)
            area = int(torch.sum(mask_tensor > 0).item())
            cluster_areas.append(area)
    else:
        for i, mask in enumerate(tqdm(all_masks, desc="Measuring cluster areas")):
            area = np.sum(mask > 0)
            cluster_areas.append(area)

    return cluster_areas, all_masks, enhanced_image

# === Process all images and return numpy array ===
def process_all_images(use_gpu=True):
    """
    Process all images in img_path_list and return cluster areas as numpy array of shape (n_images, n_classes)
    
    Args:
        use_gpu (bool): Whether to use GPU acceleration
    
    Returns:
        numpy.ndarray: Array containing cluster areas for each image
    """
    all_results = []
    total_images = sum(len(folder_images) for folder_images in img_path_list)
    
    print(f"Processing {total_images} images using {'GPU U-Net' if use_gpu and torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    image_counter = 0
    
    for folder_idx, folder_images in enumerate(img_path_list):
        folder_name = f"Folder {folder_idx + 1}"
        print(f"\n{folder_name}: Processing {len(folder_images)} images")
        print("-" * 40)
        
        for img_idx, img_path in enumerate(folder_images):
            image_counter += 1
            # print(f"\n[{image_counter}/{total_images}] Processing: {os.path.basename(img_path)}")
            
            try:
                start_time = time.time()
                
                # Get cluster areas for this image using U-Net
                cluster_areas, _, _ = get_area_gpu(img_path, use_gpu)
                
                all_results.append(cluster_areas)
                
                end_time = time.time()
                # Uncomment the next line to print completion time for each image
                # print(f"✓ Completed in {end_time - start_time:.2f} seconds")
                
            except Exception as e:
                print(f"✗ Error processing {img_path}: {str(e)}")
                # Add zeros for failed images to maintain array structure
                all_results.append(np.zeros(len(PELLET_CLASS_NAMES)))
    
    # Convert to numpy array
    results_array = np.array(all_results)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total images processed: {len(results_array)}")
    print(f"Results shape: {results_array.shape}")
    print(f"Data type: {results_array.dtype}")
    
    return results_array

# %%
# === Visualize example image with enhancements and U-Net predictions ===
example_img_path = img_path_list[0][0]  # First image from first folder
image_rgb = load_img_as_rgb(example_img_path)

print(f"Processing example image: {example_img_path}")

# Get U-Net prediction and enhanced image
pred_mask, rgb_image = predict_with_unet(example_img_path, unet_model)
cluster_areas, all_masks, _ = get_area_gpu(example_img_path, use_gpu=True)

# Create visualization
plt.figure(figsize=(24, 16))

# Original image
plt.subplot(2, 4, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

# Enhanced image
plt.subplot(2, 4, 2)
plt.imshow(rgb_image)
plt.title("Enhanced Image")
plt.axis('off')

# U-Net prediction mask (full segmentation)
plt.subplot(2, 4, 3)
# Create colored visualization of the full U-Net prediction
unet_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
colors = [
    (0, 112, 15),    # "#00700F"
    (65, 31, 31),    # "#411F1F"
    (255, 64, 64),   # "#FF4040"
    (255, 221, 26),  # "#FFDD1A"
    (117, 246, 255), # "#75F6FF"
    (128, 0, 128),   # "#800080"
    (255, 182, 193)  # "#FFB6C1"
]
if NUM_CLASSES is not None and NUM_CLASSES > len(colors):
    # Pad with neutral colors if checkpoint has more classes than predefined
    colors.extend([(200, 200, 200)] * (NUM_CLASSES - len(colors)))
class_limit = NUM_CLASSES if isinstance(NUM_CLASSES, int) else len(colors)
for class_id in range(min(class_limit, len(colors))):
    mask = pred_mask == class_id
    unet_colored[mask] = colors[class_id]
plt.imshow(unet_colored)
plt.title("U-Net Segmentation\n(All Classes)")
plt.axis('off')

# === Visualize clusters in U-Net class order ===
cluster_names = list(PELLET_CLASS_NAMES)
cluster_display = list(zip(cluster_names, all_masks, cluster_areas))

for i, (name, mask, area) in enumerate(cluster_display[:4]):
    brightness = np.mean(rgb_image[mask]) if np.any(mask) else float('nan')

    masked_img = np.zeros_like(rgb_image)
    masked_img[mask] = rgb_image[mask]

    plt.subplot(2, 4, i + 5)
    plt.imshow(masked_img)
    b_str = f"{brightness:.2f}" if not np.isnan(brightness) else "N/A"
    plt.title(f"{name}\nArea: {area} px\nBrightness: {b_str}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Print U-Net prediction statistics
print("\nU-Net Prediction Statistics:")
print("-" * 40)
unique_classes, class_counts = np.unique(pred_mask, return_counts=True)
class_name_map = {idx + 1: name for idx, name in enumerate(cluster_names)}
for class_id, count in zip(unique_classes, class_counts):
    if class_id == 0:
        print(f"Background: {count} pixels")
    else:
        class_name = class_name_map.get(class_id, f"Class {class_id}")
        print(f"Class {class_id} ({class_name}): {count} pixels")

# Print cluster statistics from U-Net
print("\nU-Net Cluster Statistics:")
print("-" * 30)
for i, (name, area) in enumerate(zip(cluster_names, cluster_areas)):
    print(f"{name}: {area} pixels")

print("\nProcessing all images to get cluster areas...")

# %%
cluster_areas_array = process_all_images(use_gpu=True)

# %%
print(cluster_areas)

# %%

def get_npk(cluster_areas):
    """Approximate NPK composition based on pellet class areas.

    Parameters:
        cluster_areas (Sequence[float]): Areas aligned with PELLET_CLASS_NAMES.
    """
    if not PELLET_CLASS_NAMES:
        raise RuntimeError("Pellet class names not initialized; load_unet_model() must be called first.")

    areas = np.asarray(cluster_areas, dtype=float)
    pellet_total = float(areas.sum())
    if pellet_total <= 0:
        return [0.0, 0.0, 0.0]

    npk_total = {'N': 0.0, 'P': 0.0, 'K': 0.0}

    for class_name, area in zip(PELLET_CLASS_NAMES, areas):
        if area <= 0:
            continue
        material_key = _material_key_for_class(class_name)
        if material_key is None:
            continue
        composition = RAW_MATERIAL_NUTRIENTS.get(material_key, {})
        fraction = area / pellet_total
        for nutrient in npk_total:
            npk_total[nutrient] += composition.get(nutrient, 0.0) * fraction

    return [round(npk_total[nutrient], 6) for nutrient in ['N', 'P', 'K']]


# Prepare data for regression
X_area = cluster_areas_array # Data preparation for regression plotting
actual_npk = [  # map NPK values from folder names
    list(map(int, os.path.basename(os.path.dirname(path)).split('-')))
    for paths in img_path_list
    for path in paths
]

# get estimated data from cluster areas for regression
approx_npk = np.array([
    get_npk(areas) for areas in X_area
])

# ============== NPK MATCHING VISUALIZATION ==============
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Convert to numpy arrays for easier manipulation
actual_npk = np.array(actual_npk)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('NPK Matching Analysis', fontsize=16, fontweight='bold')

# 1. Scatter plots for each NPK component
components = ['N', 'P', 'K']
colors = ['blue', 'red', 'green']

for i, (component, color) in enumerate(zip(components, colors)):
    ax = axes[0, i]
    
    # Scatter plot
    ax.scatter(actual_npk[:, i], approx_npk[:, i], 
               alpha=0.6, color=color, s=50, edgecolors='black', linewidth=0.5)
    
    # Perfect match line
    min_val = min(np.min(actual_npk[:, i]), np.min(approx_npk[:, i]))
    max_val = max(np.max(actual_npk[:, i]), np.max(approx_npk[:, i]))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Match')
    
    ax.set_xlabel(f'Actual {component}')
    ax.set_ylabel(f'Predicted {component}')
    ax.set_title(f'{component} Component Matching')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add correlation coefficient
    correlation = np.corrcoef(actual_npk[:, i], approx_npk[:, i])[0, 1]
    ax.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top')

# 2. Error analysis
ax = axes[1, 0]
errors = approx_npk - actual_npk
for i, (component, color) in enumerate(zip(components, colors)):
    ax.scatter(range(len(errors)), errors[:, i], 
               label=f'{component} Error', alpha=0.7, color=color, s=30)

ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax.set_xlabel('Sample Index')
ax.set_ylabel('Prediction Error')
ax.set_title('Prediction Errors by Sample')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Error distribution
ax = axes[1, 1]
ax.hist([errors[:, i] for i in range(3)], bins=15, alpha=0.7, 
        label=components, color=colors)
ax.set_xlabel('Error Value')
ax.set_ylabel('Frequency')
ax.set_title('Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Summary statistics table
ax = axes[1, 2]
ax.axis('off')

# Calculate statistics
stats_data = []
for i, component in enumerate(components):
    mae = np.mean(np.abs(errors[:, i]))
    rmse = np.sqrt(np.mean(errors[:, i]**2))
    correlation = np.corrcoef(actual_npk[:, i], approx_npk[:, i])[0, 1]
    stats_data.append([component, f'{mae:.3f}', f'{rmse:.3f}', f'{correlation:.3f}'])

# Create table
table = ax.table(cellText=stats_data,
                colLabels=['Component', 'MAE', 'RMSE', 'Correlation'],
                cellLoc='center',
                loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

# Style the table
for i in range(len(components) + 1):
    for j in range(4):
        if i == 0:  # Header row
            table[(i, j)].set_facecolor('#40466e')
            table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            table[(i, j)].set_facecolor('#f1f1f2')

ax.set_title('Performance Statistics', fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# Print detailed analysis
print("\n" + "="*50)
print("NPK MATCHING ANALYSIS SUMMARY")
print("="*50)

for i, component in enumerate(components):
    actual_vals = actual_npk[:, i]
    pred_vals = approx_npk[:, i]
    
    mae = np.mean(np.abs(pred_vals - actual_vals))
    rmse = np.sqrt(np.mean((pred_vals - actual_vals)**2))
    correlation = np.corrcoef(actual_vals, pred_vals)[0, 1]
    
    print(f"\n{component} Component:")
    print(f"  Mean Absolute Error: {mae:.3f}")
    print(f"  Root Mean Square Error: {rmse:.3f}")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  Range - Actual: {np.min(actual_vals):.1f} to {np.max(actual_vals):.1f}")
    print(f"  Range - Predicted: {np.min(pred_vals):.3f} to {np.max(pred_vals):.3f}")

# Show sample-by-sample comparison
print(f"\n{'Sample':<8} {'Actual NPK':<15} {'Predicted NPK':<20} {'Error':<15}")
print("-" * 60)
for i in range(min(10, len(actual_npk))):  # Show first 10 samples
    actual = actual_npk[i]
    predicted = approx_npk[i]
    error = predicted - actual
    print(f"{i+1:<8} {actual}<15 {predicted}<20 {error}<15")

if len(actual_npk) > 10:
    print(f"... and {len(actual_npk) - 10} more samples")

# %%
print(f"Number of cluster area samples: {len(X_area)}")
print(f"Number of actual NPK samples: {len(actual_npk)}")

# %%
for i in range(len(approx_npk)):
    print(approx_npk[i], actual_npk[i])

# %%
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import joblib

def tune_regressor(y_approx, y_true, save_path='checkpoints/regression_model.pkl', degree=2, alpha=1.0):
    """
    Train a polynomial linear regression model and save it as a checkpoint.

    Parameters:
        y_approx: Input features
        y_true: Target NPK values
        save_path: File path to save the trained model
        degree: Degree of the polynomial features
        alpha: Regularization strength for Ridge regression
    """
    # Create pipeline: Polynomial -> Ridge
    regressor = make_pipeline(
        PolynomialFeatures(degree=degree),
        Ridge(alpha=alpha)
    )

    # Train the model
    regressor.fit(y_approx, y_true)

    # Save model
    joblib.dump(regressor, save_path)
    print(f"Model checkpoint saved to: {save_path}")

    return regressor


def load_npk_regressor(model_path='checkpoints/regression_model.pkl'):
    """
    Load the saved regressor model from checkpoint.
    
    Parameters:
        model_path: Path to the saved model file
    
    Returns:
        regressor: Loaded LinearRegression model
    """
    try:
        regressor = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return regressor
    except FileNotFoundError:
        print(f"Model file not found at: {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_npk(regressor, cluster_areas):
    """
    Predict the actual NPK values based on a new cluster area input.

    Parameters:
        regressor (LinearRegression): Trained regression model
        cluster_areas (list or array): Areas aligned with PELLET_CLASS_NAMES

    Returns:
        np.ndarray: Predicted NPK values as a NumPy array [N, P, K]
    """
    # Step 1: Convert cluster area to approximate NPK using same logic
    approx_npk = np.array(get_npk(cluster_areas)).reshape(1, -1)
    
    # Step 2: Predict using the trained regression model
    predicted_npk = regressor.predict(approx_npk)

    return predicted_npk.flatten()


# Example usage:
if __name__ == "__main__":
    # tune regressor
    regressor = tune_regressor(approx_npk, actual_npk, save_path='checkpoints/regression_model.pkl')

    # Loading the saved model
    loaded_regressor = load_npk_regressor('checkpoints/regression_model.pkl')
    
    # Example prediction with loaded model
    if loaded_regressor is not None:
        for i in range(len(approx_npk)):
            predicted_npk = regressor.predict(approx_npk[i].reshape(1, -1))
            print("Improved NPK prediction:", predicted_npk.flatten(), actual_npk[i])
    else:
        print("Could not make prediction - model not loaded successfully")



# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming these variables exist:
# y_actual_npk: np.array, shape (n_samples, 3)
# X_approx_npk: np.array, shape (n_samples, 3)
# regressor: trained sklearn LinearRegression model

# Predict improved NPK using the regression model
regressor = load_npk_regressor('checkpoints/regression_model.pkl')
X_predicted_npk = regressor.predict(approx_npk)  # shape (n_samples, 3)

components = ['N', 'P', 'K']
colors = ['blue', 'red', 'green']

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('NPK Prediction Comparison: Raw Approximate vs Regression', fontsize=18, fontweight='bold')

for i, component in enumerate(components):
    # --- 1. Actual vs Raw Approximate ---
    ax = axes[0, i]
    ax.scatter(actual_npk[:, i], approx_npk[:, i], 
               alpha=0.6, color=colors[i], s=50, edgecolors='black', linewidth=0.5)
    min_val = min(np.min(actual_npk[:, i]), np.min(approx_npk[:, i]))
    max_val = max(np.max(actual_npk[:, i]), np.max(approx_npk[:, i]))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    ax.set_xlabel(f'Actual {component}')
    ax.set_ylabel(f'Raw Approximate {component}')
    ax.set_title(f'{component}: Actual vs Raw Approximate')
    ax.grid(True, alpha=0.3)

    # --- 2. Actual vs Regression Predicted ---
    ax = axes[1, i]
    ax.scatter(actual_npk[:, i], X_predicted_npk[:, i], 
               alpha=0.6, color=colors[i], s=50, edgecolors='black', linewidth=0.5)
    min_val = min(np.min(actual_npk[:, i]), np.min(X_predicted_npk[:, i]))
    max_val = max(np.max(actual_npk[:, i]), np.max(X_predicted_npk[:, i]))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    ax.set_xlabel(f'Actual {component}')
    ax.set_ylabel(f'Regression Predicted {component}')
    ax.set_title(f'{component}: Actual vs Regression Predicted')
    ax.grid(True, alpha=0.3)

    # --- 3. Error Comparison (Raw vs Regression) ---
    ax = axes[2, i]
    raw_errors = approx_npk[:, i] - actual_npk[:, i]
    reg_errors = X_predicted_npk[:, i] - actual_npk[:, i]
    ax.scatter(range(len(raw_errors)), raw_errors, label='Raw Error', alpha=0.7, color='gray', s=30)
    ax.scatter(range(len(reg_errors)), reg_errors, label='Regression Error', alpha=0.7, color=colors[i], s=30)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Error')
    ax.set_title(f'{component}: Prediction Errors Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- Print summary stats for raw and regression ---

def print_stats(errors, label):
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    print(f"{label} MAE: {mae:.3f}, RMSE: {rmse:.3f}")

print("=== Error statistics comparison ===")
for i, component in enumerate(components):
    print(f"\nComponent: {component}")
    raw_errors = approx_npk[:, i] - actual_npk[:, i]
    reg_errors = X_predicted_npk[:, i] - actual_npk[:, i]
    print_stats(raw_errors, "Raw Approximate")
    print_stats(reg_errors, "Regression Predicted")
