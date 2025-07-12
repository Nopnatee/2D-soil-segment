import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import time
import joblib

# Import your custom U-Net
from custom_unet import SimpleUNet, ConvBlock


class NPKPredictor:
    """
    A complete NPK prediction system using U-Net for segmentation and 
    regression model for NPK value estimation.
    """
    
    def __init__(self, 
                 unet_model_path="checkpoints/best_model.pth",
                 regression_model_path="checkpoints/regression_model.pkl",
                 num_classes=5,
                 bead_masks=4):
        """
        Initialize the NPK Predictor.
        
        Args:
            unet_model_path (str): Path to the trained U-Net model
            regression_model_path (str): Path to the trained regression model
            num_classes (int): Number of classes in U-Net output
            bead_masks (int): Number of bead mask classes to extract
        """
        self.NUM_CLASSES = num_classes
        self.BEAD_MASKS = bead_masks
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.DEVICE}")
        
        # Load models
        self.unet_model = self._load_unet_model(unet_model_path)
        self.regressor = self._load_regression_model(regression_model_path)
        
        # Initialize transforms
        self.unet_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # must match training size
            transforms.ToTensor(),          # Converts to tensor [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # same normalization as training
        ])
        
        # Define bead compositions for approximate NPK calculation
        self.bead_compositions = [
            {'N': 18, 'P': 46, 'K': 0},   # black
            {'N': 0, 'P': 0, 'K': 60},   # red
            {'N': 21, 'P': 0, 'K': 0},   # stain
            {'N': 46, 'P': 0, 'K': 0}   # white
        ]
    
    def _load_unet_model(self, model_path):
        """Load the trained U-Net model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.DEVICE)
            model = SimpleUNet(in_channels=3, n_classes=self.NUM_CLASSES)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to(self.DEVICE)
            print(f"U-Net model loaded from: {model_path}")
            return model
        except Exception as e:
            print(f"Error loading U-Net model: {e}")
            return None
    
    def _load_regression_model(self, model_path):
        """Load the saved regressor model"""
        try:
            regressor = joblib.load(model_path)
            print(f"Regression model loaded from: {model_path}")
            return regressor
        except FileNotFoundError:
            print(f"Regression model file not found at: {model_path}")
            return None
        except Exception as e:
            print(f"Error loading regression model: {e}")
            return None
    
    def _load_img_as_rgb(self, img_path):
        """Load image and convert to RGB"""
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found at: {img_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    def _predict_with_unet(self, image_path):
        """
        Use U-Net to predict segmentation masks
        
        Args:
            image_path (str): Path to the input image
        
        Returns:
            numpy.ndarray: Predicted segmentation mask
            numpy.ndarray: Enhanced image for visualization
        """
        if self.unet_model is None:
            raise ValueError("U-Net model not loaded properly")
        
        # Load and enhance the image
        rgb_image = self._load_img_as_rgb(image_path)
        
        # Convert enhanced image to PIL for U-Net preprocessing
        enhanced_pil = Image.fromarray(rgb_image.astype(np.uint8))
        
        # Preprocess for U-Net
        input_tensor = self.unet_transform(enhanced_pil).unsqueeze(0).to(self.DEVICE)
        
        # Get U-Net prediction
        with torch.no_grad():
            output = self.unet_model(input_tensor)  # Shape: [1, num_classes, H, W]
        
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
    
    def get_cluster_masks(self, image_path, use_gpu=True):
        """
        Get cluster masks using U-Net prediction
        
        Args:
            image_path (str): Path to the input image
            use_gpu (bool): Whether to use GPU acceleration
        
        Returns:
            list: List of cluster masks
            numpy.ndarray: Enhanced image
        """
        if not use_gpu or not torch.cuda.is_available():
            raise NotImplementedError("GPU acceleration is required for this function.")
        
        # Get U-Net prediction
        pred_mask, enhanced_image = self._predict_with_unet(image_path)
        
        H, W = pred_mask.shape
        
        # Create individual masks for each class (excluding background class 0)
        all_masks = []
        
        # Assume background is class 0, so we extract classes 1 to BEAD_MASKS
        for class_id in range(1, min(self.NUM_CLASSES, self.BEAD_MASKS + 1)):
            class_mask = (pred_mask == class_id)
            all_masks.append(class_mask)
        
        # If we have fewer classes than expected, pad with empty masks
        while len(all_masks) < self.BEAD_MASKS:
            all_masks.append(np.zeros((H, W), dtype=bool))
        
        # Only keep the first BEAD_MASKS classes
        all_masks = all_masks[:self.BEAD_MASKS]
        
        # Compute average brightness for each mask to maintain sorting behavior
        brightness_averages = []
        enhanced_tensor = torch.from_numpy(enhanced_image).to(self.DEVICE)
        
        for mask in all_masks:
            mask_tensor = torch.from_numpy(mask).to(self.DEVICE)
            masked_pixels = enhanced_tensor[mask_tensor]
            if len(masked_pixels) == 0:
                brightness = 0
            else:
                brightness = float(torch.mean(masked_pixels.float()))
            brightness_averages.append(brightness)
        
        # Get sorted indices from brightest to darkest
        sorted_indices = np.argsort(brightness_averages)[::-1]
        
        # Reorder clusters by brightness (maintaining original behavior)
        all_masks = [all_masks[i] for i in sorted_indices]
        
        return all_masks, enhanced_image
    
    def get_cluster_areas(self, image_path, use_gpu=True):
        """
        Get cluster areas from segmentation masks
        
        Args:
            image_path (str): Path to the input image
            use_gpu (bool): Whether to use GPU acceleration
        
        Returns:
            list: List of cluster areas
        """
        all_masks, rgb_image = self.get_cluster_masks(image_path, use_gpu)
        
        # Compute cluster areas
        cluster_areas = []
        if use_gpu and torch.cuda.is_available():
            for i, mask in enumerate(tqdm(all_masks, desc="Measuring cluster areas")):
                mask_tensor = torch.from_numpy(mask).to(self.DEVICE)
                area = int(torch.sum(mask_tensor > 0).item())
                cluster_areas.append(area)
        else:
            for i, mask in enumerate(tqdm(all_masks, desc="Measuring cluster areas")):
                area = np.sum(mask > 0)
                cluster_areas.append(area)

        return cluster_areas
    
    def get_approximate_npk(self, cluster_areas):
        """
        Get approximate NPK values based on cluster areas using predefined compositions
        
        Args:
            cluster_areas (list or array): [white, stain, red, black] areas
        
        Returns:
            list: [N, P, K] values
        """
        npk_total = {'N': 0, 'P': 0, 'K': 0}
        
        for i, area in enumerate(cluster_areas):
            if i < len(self.bead_compositions):
                for key in npk_total:
                    npk_total[key] += self.bead_compositions[i][key] * area
        
        total_beads = sum(cluster_areas)
        if total_beads == 0:
            return [0, 0, 0]
        
        return [round(npk_total[key] / total_beads, 2) for key in ['N', 'P', 'K']]
    
    def predict_npk(self, cluster_areas):
        """
        Predict the actual NPK values using the trained regression model
        
        Args:
            cluster_areas (list or array): [white, stain, red, black] areas
        
        Returns:
            numpy.ndarray: Predicted NPK values as [N, P, K]
        """
        if self.regressor is None:
            print("Warning: Regression model not loaded. Using approximate NPK calculation.")
            return np.array(self.get_approximate_npk(cluster_areas))
        
        # Step 1: Convert cluster area to approximate NPK using same logic
        approx_npk = np.array(self.get_approximate_npk(cluster_areas)).reshape(1, -1)
        
        # Step 2: Predict using the trained regression model
        predicted_npk = self.regressor.predict(approx_npk)
        
        return predicted_npk.flatten()
    
    def predict_from_image(self, image_path, use_gpu=True):
        """
        Complete NPK prediction pipeline from image
        
        Args:
            image_path (str): Path to the input image
            use_gpu (bool): Whether to use GPU acceleration
        
        Returns:
            dict: Dictionary containing cluster areas and NPK predictions
        """
        try:
            # Get cluster areas
            cluster_areas = self.get_cluster_areas(image_path, use_gpu)
            
            # Get approximate NPK
            approx_npk = self.get_approximate_npk(cluster_areas)
            
            # Get improved NPK prediction
            predicted_npk = self.predict_npk(cluster_areas)
            
            results = {
                'cluster_areas': cluster_areas,
                'approximate_npk': approx_npk,
                'predicted_npk': predicted_npk.tolist(),
                'image_path': image_path
            }
            
            return results
            
        except Exception as e:
            print(f"Error in prediction pipeline: {e}")
            return None
    
    def print_results(self, results):
        """
        Print prediction results in a formatted way
        
        Args:
            results (dict): Results from predict_from_image
        """
        if results is None:
            print("No results to display")
            return
        
        print(f"\n=== NPK Prediction Results ===")
        print(f"Image: {results['image_path']}")
        print(f"Cluster Areas: {results['cluster_areas']}")
        print(f"Approximate NPK: N={results['approximate_npk'][0]}, P={results['approximate_npk'][1]}, K={results['approximate_npk'][2]}")
        print(f"Predicted NPK: N={results['predicted_npk'][0]:.2f}, P={results['predicted_npk'][1]:.2f}, K={results['predicted_npk'][2]:.2f}")


# Example usage:
if __name__ == "__main__":
    # Initialize the predictor
    predictor = NPKPredictor(
        unet_model_path="checkpoints/best_model.pth",
        regression_model_path="checkpoints/npk_regressor.pkl"
    )
    
    # Make prediction
    image_path = "pictures/15-4-20/IMG_0869.jpg"
    results = predictor.predict_from_image(image_path, use_gpu=True)
    
    # Print results
    predictor.print_results(results)
    
    # Or access individual components
    if results:
        print(f"\nCluster areas: {results['cluster_areas']}")
        print(f"Final NPK prediction: {results['predicted_npk']}")