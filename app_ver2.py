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
            transforms.Resize((256, 256)),
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
        
        # Updated color map to match the reference image
        self.color_map = {
            0: [0, 0, 0],    # Background - brown/dark brown (matches image)
            1: [139, 69, 19],      # Class 1 
            2: [255, 0, 0],    # Class 2 
            3: [255, 255, 0],    # Class 3 
            4: [0, 255, 255],      # Class 4 
        }
        
        # Class names for better labeling
        self.class_names = {
            0: 'Background',
            1: 'black',
            2: 'red', 
            3: 'stain',
            4: 'white'
        }
    
    def _load_models(self, unet_path, regression_path):
        """Load both U-Net and regression models."""
        try:
            # Load U-Net
            checkpoint = torch.load(unet_path, map_location=self.DEVICE)
            unet_model = SimpleUNet(in_channels=3, n_classes=self.NUM_CLASSES)
            unet_model.load_state_dict(checkpoint['model_state_dict'])
            unet_model.eval().to(self.DEVICE)
            print(f"U-Net model loaded from: {unet_path} as {type(unet_model)}")
            
            # Load regression model
            self.regressor = joblib.load(regression_path)
            print(f"Regression model loaded from: {regression_path} as {type(self.regressor)}")
            
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
    
    def create_colored_segmentation_mask(self, pred_mask):
        """Create colored segmentation mask matching the reference image style."""
        colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        
        for class_id, color in self.color_map.items():
            mask = pred_mask == class_id
            colored_mask[mask] = color
        
        return colored_mask
    
    def create_visualization_plots(self, image_array, pred_mask, cluster_areas, approx_npk, predicted_npk):
        """Create comprehensive visualization plots with matching color scheme."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('NPK Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Original Image
        axes[0, 0].imshow(image_array)
        axes[0, 0].set_title('Original Image', fontsize=14)
        axes[0, 0].axis('off')
        
        # 2. Predicted Segmentation (matching reference image style)
        colored_mask = self.create_colored_segmentation_mask(pred_mask)
        axes[0, 1].imshow(colored_mask)
        axes[0, 1].set_title('Predicted Segmentation', fontsize=14)
        axes[0, 1].axis('off')
        
        # 3. Overlay with improved blending
        overlay = cv2.addWeighted(image_array.astype(np.uint8), 0.6, colored_mask, 0.4, 0)
        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title('Overlay', fontsize=14)
        axes[0, 2].axis('off')
        
        # 4. Class Distribution (matching reference image style)
        class_counts = []
        class_labels = []
        colors_for_bars = []
        
        total_pixels = pred_mask.size
        
        for class_id in range(self.NUM_CLASSES):
            count = np.sum(pred_mask == class_id)
            percentage = (count / total_pixels) * 100
            
            class_counts.append(percentage)
            class_labels.append(self.class_names.get(class_id, f'Class {class_id}'))
            
            # Convert RGB to matplotlib color (0-1 range)
            color_rgb = [c/255.0 for c in self.color_map[class_id]]
            colors_for_bars.append(color_rgb)
        
        bars = axes[1, 0].bar(class_labels, class_counts, color=colors_for_bars, alpha=0.8)
        axes[1, 0].set_title('Class Distribution', fontsize=14)
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add percentage labels on bars
        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(class_counts)*0.01,
                           f'{count:.1f}%', ha='center', va='bottom', fontsize=10)
        
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
        
        # 6. Enhanced Model Summary (matching reference image style)
        axes[1, 2].axis('off')
        
        # Calculate image size and total pixels
        img_height, img_width = pred_mask.shape
        total_pixels = img_height * img_width
        
        # Create comprehensive summary
        summary_text = f"Image Size: ({img_width}, {img_height})\n"
        summary_text += f"Total Pixels: {total_pixels:,}\n\n"
        
        summary_text += "Class Distribution:\n"
        for class_id in range(self.NUM_CLASSES):
            count = np.sum(pred_mask == class_id)
            percentage = (count / total_pixels) * 100
            class_name = self.class_names.get(class_id, f'Class {class_id}')
            summary_text += f"{class_name}: {percentage:.1f}%\n"
        
        summary_text += "\nBead Compositions:\n"
        for i, comp in enumerate(self.bead_compositions):
            if i < len(cluster_areas):
                summary_text += f"{comp['name']}:\n"
                summary_text += f"  N: {comp['N']}% | P: {comp['P']}% | K: {comp['K']}%\n"
                summary_text += f"  Area: {cluster_areas[i]:,} pixels\n"
        
        # Create a styled text box
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[1, 2].set_title('Model Summary', fontsize=14)
        
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
        
        return "\n".join(results)

# Initialize the predictor
predictor = NPKPredictorGradio(
    unet_model_path="checkpoints/best_model.pth",
    regression_model_path="checkpoints/regression_model.pkl"
)

# Create custom CSS for styling
custom_css = """
/* Import Fira Code font */
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600;700&display=swap');

/* Set global font and dark theme */
* {
    font-family: 'Fira Code', monospace !important;
}

/* Main container styling - Dark theme */
.gradio-container {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
}

body {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
}

/* Input section styling - Dark boxes */
.input-section {
    background-color: #2d2d2d !important;
    border: 1px solid #404040 !important;
    border-radius: 8px !important;
    padding: 16px !important;
    margin: 8px !important;
}

/* Results section styling - Dark boxes */
.results-section {
    background-color: #2d2d2d !important;
    border: 1px solid #404040 !important;
    border-radius: 8px !important;
    padding: 16px !important;
    margin: 8px !important;
}

/* Model status styling */
.model-status {
    background-color: #2d2d2d !important;
    border: 1px solid #404040 !important;
    border-radius: 8px !important;
    padding: 16px !important;
    margin: 16px 0 !important;
}

/* Orange button styling */
.orange-btn {
    background: linear-gradient(45deg, #ff6b35, #ff8c42) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.orange-btn:hover {
    background: linear-gradient(45deg, #ff5722, #ff7043) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3) !important;
}

/* Textbox styling - Dark theme */
.gr-textbox {
    background-color: #3a3a3a !important;
    border: 1px solid #555555 !important;
    border-radius: 6px !important;
    color: #ffffff !important;
    font-family: 'Fira Code', monospace !important;
}

/* Image component styling - Dark theme */
.gr-image {
    background-color: #3a3a3a !important;
    border: 1px solid #555555 !important;
    border-radius: 6px !important;
}

/* Markdown styling - Dark theme */
.gr-markdown {
    color: #ffffff !important;
    font-family: 'Fira Code', monospace !important;
}

/* Header styling - Dark theme */
.gr-markdown h1 {
    color: #ffffff !important;
    font-weight: 700 !important;
    margin-bottom: 16px !important;
}

.gr-markdown h2 {
    color: #e0e0e0 !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
}

.gr-markdown h3 {
    color: #cccccc !important;
    font-weight: 500 !important;
    margin-bottom: 8px !important;
}

/* Checkbox styling */
.gr-checkbox {
    background-color: #3a3a3a !important;
    border: 1px solid #555555 !important;
    color: #ffffff !important;
}

/* Label styling */
.gr-label {
    color: #ffffff !important;
    font-family: 'Fira Code', monospace !important;
}

/* Success message styling */
.success-message {
    background-color: #2d4a2d !important;
    border: 1px solid #4a8c4a !important;
    border-radius: 6px !important;
    color: #90ee90 !important;
    padding: 12px !important;
    margin: 8px 0 !important;
}

/* Custom spacing */
.section-spacing {
    margin-bottom: 24px !important;
}

/* Ensure proper dark theme for all components */
.gr-block {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
}

/* Tab styling */
.gr-tab-nav {
    background-color: #2d2d2d !important;
    border-bottom: 1px solid #404040 !important;
}

/* Button group styling */
.gr-button-group {
    gap: 8px !important;
}

/* File upload styling */
.gr-file-upload {
    background-color: #3a3a3a !important;
    border: 1px solid #555555 !important;
    border-radius: 6px !important;
    color: #ffffff !important;
}
"""

# Create Gradio interface
def create_gradio_interface():
    """Create the Gradio interface with custom styling."""
    
    with gr.Blocks(
        title="NPK Soil Analysis System",
        css=custom_css,
        theme=gr.themes.Base()
    ) as demo:
        # Title and main description
        gr.Markdown("""
        # 🌱 NPK Soil Analysis System
        
        Upload an image of soil beads to analyze NPK (Nitrogen, Phosphorus, Potassium) content. The system uses a U-Net model for bead segmentation and machine learning for NPK prediction.
        """, elem_classes=["section-spacing"])
        
        # Instructions section
        gr.Markdown("""
        **Instructions:**
        1. Upload a clear image of soil beads
        2. Choose whether to use GPU acceleration (if available)
        3. Click 'Analyze NPK' to get results
        """, elem_classes=["section-spacing"])
        
        # Model Status
        gr.Markdown("""
        ### Model Status
        ✅ **Models loaded successfully!**
        """, elem_classes=["model-status"])
        
        # Main interface with two columns
        with gr.Row():
            # Input Column
            with gr.Column(scale=1, elem_classes=["input-section"]):
                gr.Markdown("### 📂 Input")
                image_input = gr.Image(
                    label="Upload Soil Bead Image",
                    type="numpy",
                    height=400,
                    container=True
                )
                
                # GPU acceleration checkbox
                gpu_checkbox = gr.Checkbox(
                    label="Use GPU Acceleration",
                    value=torch.cuda.is_available(),
                    info="Enable if you have a CUDA-compatible GPU",
                    container=True
                )
                
                predict_btn = gr.Button(
                    "🔬 Analyze NPK", 
                    variant="primary",
                    size="lg",
                    elem_classes=["orange-btn"]
                )
                
                # Example images section
                gr.Markdown("### 📷 Example Images")
                gr.Markdown("≡ Try these examples", elem_classes=["section-spacing"])
            
            # Results Column
            with gr.Column(scale=2, elem_classes=["results-section"]):
                # Results and Analysis tabs
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Results")
                        
                        # Text results
                        results_output = gr.Textbox(
                            label="NPK Analysis Results",
                            lines=20,
                            max_lines=25,
                            show_copy_button=True,
                            container=True,
                            interactive=False,
                            value="Click 'Analyze NPK' to see results here..."
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📈 Analysis Visualization")
                        
                        # Visualization output
                        viz_output = gr.Image(
                            label="Analysis Visualization",
                            type="pil",
                            height=500,
                            container=True
                        )
        
        # Event handlers
        predict_btn.click(
            fn=predictor.predict_from_image,
            inputs=[image_input],
            outputs=[viz_output, results_output],
            show_progress=True
        )
        
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
