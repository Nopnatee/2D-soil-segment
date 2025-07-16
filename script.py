import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import joblib

# Import your custom U-Net
from custom_unet import SimpleUNet, ConvBlock


class NPKPredictor:
    """Simplified NPK prediction system using U-Net for segmentation and regression model for NPK estimation."""
    
    def __init__(self, 
                 unet_model_path="checkpoints/best_model.pth",
                 regression_model_path="checkpoints/regression_model.pkl",
                 num_classes=5):
        """Initialize the NPK Predictor with model paths."""
        self.NUM_CLASSES = num_classes
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.DEVICE}")
        
        # Load models
        self.unet_model = self._load_models(unet_model_path, regression_model_path)
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Bead compositions for NPK calculation
        self.bead_compositions = [
            {'N': 18, 'P': 46, 'K': 0},   # black
            {'N': 0, 'P': 0, 'K': 60},   # red  
            {'N': 21, 'P': 0, 'K': 0},   # stain
            {'N': 46, 'P': 0, 'K': 0}    # white
        ]
    
    def _load_models(self, unet_path, regression_path):
        """Load both U-Net and regression models."""
        try:
            # Load U-Net
            checkpoint = torch.load(unet_path, map_location=self.DEVICE)
            unet_model = SimpleUNet(in_channels=3, n_classes=self.NUM_CLASSES)
            unet_model.load_state_dict(checkpoint['model_state_dict'])
            unet_model.eval().to(self.DEVICE)
            print(f"U-Net model loaded from: {unet_path}")
            
            # Load regression model
            self.regressor = joblib.load(regression_path)
            print(f"Regression model loaded from: {regression_path}")
            
            return unet_model
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.regressor = None
            return None
    
    def _get_segmentation_mask(self, image_path):
        """Get segmentation mask from image using U-Net."""
        if self.unet_model is None:
            raise ValueError("U-Net model not loaded properly")
        
        # Load and preprocess image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")
        
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image.astype(np.uint8))
        
        # Transform and predict
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.DEVICE)
        
        with torch.no_grad():
            output = self.unet_model(input_tensor)
        
        # Get prediction mask
        if output.shape[1] == 1:
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
        else:
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Resize back to original size
        original_size = rgb_image.shape[:2]
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                              (original_size[1], original_size[0]), 
                              interpolation=cv2.INTER_NEAREST)
        
        return pred_mask, rgb_image
    
    def get_cluster_areas(self, image_path):
        """Get cluster areas from segmentation masks."""
        pred_mask, rgb_image = self._get_segmentation_mask(image_path)
        
        # Extract masks for each class (excluding background class 0)
        cluster_areas = []
        enhanced_tensor = torch.from_numpy(rgb_image).to(self.DEVICE)
        
        masks_with_index = []
        
        # Get masks for classes 1 to 4 (assuming 4 bead types)
        for class_id in range(1, min(self.NUM_CLASSES, 5)):
            class_mask = (pred_mask == class_id)
            
            # Store mask with its predicted class index
            masks_with_index.append((class_mask, class_id))
        
        # Sort by predicted class index (ascending order: 1, 2, 3, 4)
        masks_with_index.sort(key=lambda x: x[1])
        
        # Calculate areas in the order of predicted indices
        for mask, class_id in masks_with_index:
            if torch.cuda.is_available():
                mask_tensor = torch.from_numpy(mask).to(self.DEVICE)
                area = int(torch.sum(mask_tensor).item())
            else:
                area = int(np.sum(mask))
            cluster_areas.append(area)
        
        # Pad with zeros if we have fewer than 4 classes
        while len(cluster_areas) < 4:
            cluster_areas.append(0)
        
        return cluster_areas[:4]  # Return only first 4
    
    def calculate_npk(self, cluster_areas):
        """Calculate NPK values from cluster areas."""
        # Approximate NPK calculation
        npk_total = {'N': 0, 'P': 0, 'K': 0}
        
        for i, area in enumerate(cluster_areas):
            if i < len(self.bead_compositions):
                for key in npk_total:
                    npk_total[key] += self.bead_compositions[i][key] * area
        
        total_beads = sum(cluster_areas)
        if total_beads == 0:
            return [0, 0, 0], [0, 0, 0]
        
        approx_npk = [round(npk_total[key] / total_beads, 6) for key in ['N', 'P', 'K']]
        
        # Predict using regression model if available
        if self.regressor is not None:
            predicted_npk = self.regressor.predict(np.array(approx_npk).reshape(1, -1))
            return approx_npk, predicted_npk.flatten().tolist()
        else:
            return approx_npk, approx_npk
    
    def predict_from_image(self, image_path):
        """Complete NPK prediction pipeline from image."""
        try:
            # Get cluster areas
            cluster_areas = self.get_cluster_areas(image_path)
            
            # Calculate NPK values
            approx_npk, predicted_npk = self.calculate_npk(cluster_areas)
            
            results = {
                'cluster_areas': cluster_areas,
                'approximate_npk': approx_npk,
                'predicted_npk': predicted_npk,
                'image_path': image_path
            }
            
            # Print results
            self.print_results(results)
            return results
            
        except Exception as e:
            print(f"Error in prediction pipeline: {e}")
            return None
    
    def print_results(self, results):
        """Print prediction results."""
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
        regression_model_path="checkpoints/regression_model.pkl"
    )
    
    # Make prediction
    image_path = "regressor_dataset/15-4-20/IMG_0869.jpg"
    results = predictor.predict_from_image(image_path)
    
    # Access results if needed
    if results:
        print(f"\nCluster areas: {results['cluster_areas']}")
        print(f"Final NPK prediction: {results['predicted_npk']}")