import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import io
import base64

# Import your custom U-Net
from soil_segment.custom_unet import SimpleUNet, ConvBlock

class NPKPredictorGradio:
    """Enhanced NPK prediction system with visualization for Gradio interface."""
    
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
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Bead compositions for NPK calculation
        self.bead_compositions = [
            {'N': 18, 'P': 46, 'K': 0, 'color': 'black', 'name': 'Black Beads'},   # black
            {'N': 0, 'P': 0, 'K': 60, 'color': 'red', 'name': 'Red Beads'},       # red  
            {'N': 21, 'P': 0, 'K': 0, 'color': 'brown', 'name': 'Brown Beads'},   # stain
            {'N': 46, 'P': 0, 'K': 0, 'color': 'white', 'name': 'White Beads'}    # white
        ]
        
        # Color map for visualization
        self.color_map = {
            0: [0, 0, 0],        # background - black
            1: [255, 0, 0],      # class 1 - red
            2: [0, 255, 0],      # class 2 - green
            3: [0, 0, 255],      # class 3 - blue
            4: [255, 255, 0],    # class 4 - yellow
        }
    
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
    
    def _get_segmentation_mask(self, image_array):
        """Get segmentation mask from image array using U-Net."""
        if self.unet_model is None:
            raise ValueError("U-Net model not loaded properly")
        
        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            rgb_image = image_array
        else:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
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
    
    def create_visualization_plots(self, image_array, pred_mask, cluster_areas, approx_npk, predicted_npk):
        """Create comprehensive visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('NPK Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Original Image
        axes[0, 0].imshow(image_array)
        axes[0, 0].set_title('Original Image', fontsize=14)
        axes[0, 0].axis('off')
        
        # 2. Segmentation Mask
        colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for class_id, color in self.color_map.items():
            colored_mask[pred_mask == class_id] = color
        
        axes[0, 1].imshow(colored_mask)
        axes[0, 1].set_title('Segmentation Mask', fontsize=14)
        axes[0, 1].axis('off')
        
        # 3. Overlay
        overlay = cv2.addWeighted(image_array.astype(np.uint8), 0.7, colored_mask, 0.3, 0)
        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title('Segmentation Overlay', fontsize=14)
        axes[0, 2].axis('off')
        
        # 4. Cluster Areas Bar Chart
        bead_names = [comp['name'] for comp in self.bead_compositions]
        colors = [comp['color'] for comp in self.bead_compositions]
        
        bars = axes[1, 0].bar(bead_names, cluster_areas, color=colors, alpha=0.7)
        axes[1, 0].set_title('Cluster Areas (pixels)', fontsize=14)
        axes[1, 0].set_ylabel('Area (pixels)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, area in zip(bars, cluster_areas):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(cluster_areas)*0.01,
                           f'{area}', ha='center', va='bottom', fontsize=10)
        
        # 5. NPK Comparison
        npk_labels = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
        x_pos = np.arange(len(npk_labels))
        width = 0.35
        
        bars1 = axes[1, 1].bar(x_pos - width/2, approx_npk, width, 
                              label='Approximate NPK', alpha=0.7, color='skyblue')
        bars2 = axes[1, 1].bar(x_pos + width/2, predicted_npk, width, 
                              label='Predicted NPK', alpha=0.7, color='lightcoral')
        
        axes[1, 1].set_title('NPK Values Comparison', fontsize=14)
        axes[1, 1].set_ylabel('NPK Value')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(npk_labels)
        axes[1, 1].legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(max(approx_npk), max(predicted_npk))*0.01,
                               f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Bead Composition Information
        axes[1, 2].axis('off')
        info_text = "Bead Compositions:\n\n"
        for i, comp in enumerate(self.bead_compositions):
            info_text += f"{comp['name']}:\n"
            info_text += f"  N: {comp['N']}%\n"
            info_text += f"  P: {comp['P']}%\n"
            info_text += f"  K: {comp['K']}%\n"
            info_text += f"  Area: {cluster_areas[i]} pixels\n\n"
        
        axes[1, 2].text(0.1, 0.9, info_text, transform=axes[1, 2].transAxes, 
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        axes[1, 2].set_title('Bead Information', fontsize=14)
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return Image.open(buf)
    
    def get_cluster_areas(self, image_array):
        """Get cluster areas from segmentation masks."""
        pred_mask, rgb_image = self._get_segmentation_mask(image_array)
        
        cluster_areas = []
        masks_with_index = []
        
        # Get masks for classes 1 to 4 (assuming 4 bead types)
        for class_id in range(1, min(self.NUM_CLASSES, 5)):
            class_mask = (pred_mask == class_id)
            masks_with_index.append((class_mask, class_id))
        
        # Sort by predicted class index
        masks_with_index.sort(key=lambda x: x[1])
        
        # Calculate areas
        for mask, class_id in masks_with_index:
            area = int(np.sum(mask))
            cluster_areas.append(area)
        
        # Pad with zeros if we have fewer than 4 classes
        while len(cluster_areas) < 4:
            cluster_areas.append(0)
        
        return cluster_areas[:4], pred_mask, rgb_image
    
    def calculate_npk(self, cluster_areas):
        """Calculate NPK values from cluster areas."""
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
    
    def predict_from_image(self, image_array):
        """Complete NPK prediction pipeline from image array."""
        try:
            if image_array is None:
                return None, "Please upload an image first."
            
            # Get cluster areas and segmentation
            cluster_areas, pred_mask, rgb_image = self.get_cluster_areas(image_array)
            
            # Calculate NPK values
            approx_npk, predicted_npk = self.calculate_npk(cluster_areas)
            
            # Create visualization
            viz_image = self.create_visualization_plots(
                rgb_image, pred_mask, cluster_areas, approx_npk, predicted_npk
            )
            
            # Create detailed results text
            results_text = self.format_results(cluster_areas, approx_npk, predicted_npk)
            
            return viz_image, results_text
            
        except Exception as e:
            error_msg = f"Error in prediction pipeline: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def format_results(self, cluster_areas, approx_npk, predicted_npk):
        """Format results as detailed text."""
        results = []
        results.append("=== NPK PREDICTION RESULTS ===\n")
        
        # Cluster areas
        results.append("📊 CLUSTER AREAS (pixels):")
        for i, (area, comp) in enumerate(zip(cluster_areas, self.bead_compositions)):
            results.append(f"  • {comp['name']}: {area:,} pixels")
        results.append(f"  • Total: {sum(cluster_areas):,} pixels\n")
        
        # Bead compositions
        results.append("🔬 BEAD COMPOSITIONS:")
        for comp in self.bead_compositions:
            results.append(f"  • {comp['name']}: N={comp['N']}%, P={comp['P']}%, K={comp['K']}%")
        results.append("")
        
        # NPK calculations
        results.append("🧪 NPK CALCULATIONS:")
        results.append(f"  • Approximate NPK: N={approx_npk[0]:.4f}, P={approx_npk[1]:.4f}, K={approx_npk[2]:.4f}")
        results.append(f"  • Predicted NPK:   N={predicted_npk[0]:.4f}, P={predicted_npk[1]:.4f}, K={predicted_npk[2]:.4f}")
        results.append("")
        
        dominant_nutrient = ['Nitrogen', 'Phosphorus', 'Potassium'][np.argmax(predicted_npk)]
        results.append(f"  • Dominant nutrient: {dominant_nutrient}")
        
        return "\n".join(results)

# Initialize the predictor
predictor = NPKPredictorGradio(
    unet_model_path="checkpoints/best_model.pth",
    regression_model_path="checkpoints/regression_model.pkl"
)

# Create Gradio interface
def create_gradio_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="NPK Prediction System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🌱 NPK Prediction System
        
        Upload an image of soil test beads to predict NPK (Nitrogen, Phosphorus, Potassium) values.
        The system uses U-Net for segmentation and regression models for accurate NPK prediction.
        
        ## How it works:
        1. **Image Upload**: Upload your soil test bead image
        2. **Segmentation**: U-Net model segments different bead types
        3. **Area Calculation**: Calculate pixel areas for each bead type
        4. **NPK Prediction**: Predict NPK values using bead compositions and regression model
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📤 Input")
                image_input = gr.Image(
                    label="Upload Soil Test Bead Image",
                    type="numpy",
                    height=300
                )
                
                predict_btn = gr.Button(
                    "🔍 Analyze Image", 
                    variant="primary",
                    size="lg"
                )
                
                # Model info
                gr.Markdown("""
                ### ℹ️ Model Information
                - **Segmentation**: U-Net with 5 classes
                - **Regression**: Trained on soil test data
                - **Bead Types**: Black, Red, Brown, White
                - **Device**: GPU if available, else CPU
                """)
            
            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### 📊 Results")
                
                # Visualization output
                viz_output = gr.Image(
                    label="Analysis Visualization",
                    type="pil",
                    height=600
                )
                
                # Text results
                results_output = gr.Textbox(
                    label="Detailed Results",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True,
                    container=True
                )
        
        # Event handlers
        predict_btn.click(
            fn=predictor.predict_from_image,
            inputs=[image_input],
            outputs=[viz_output, results_output],
            show_progress=True
        )
        
        # Example section
        gr.Markdown("""
        ### 📝 Instructions:
        1. Take a clear photo of your soil test beads
        2. Ensure good lighting and contrast
        3. Upload the image using the interface above
        4. Click "Analyze Image" to get NPK predictions
        5. View the detailed visualization and results
        
        ### 🎯 Expected Results:
        - Segmentation mask showing different bead types
        - Pixel area calculations for each bead type
        - NPK value predictions with confidence metrics
        - Detailed analysis and recommendations
        """)
        
        # Footer
        gr.Markdown("""
        ---
        **Note**: This system is designed for educational and research purposes. 
        For critical agricultural decisions, please consult with soil testing professionals.
        """)
    
    return demo

# Launch the application
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        inbrowser=True
    )
