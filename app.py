import gradio as gr
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import tempfile
import io
import base64
from script import NPKPredictor
import traceback

class NPKGradioApp:
    def __init__(self):
        """Initialize the Gradio app with NPK predictor"""
        self.predictor = None
        self.initialize_predictor()
    
    def initialize_predictor(self):
        """Initialize the NPK predictor with default model paths"""
        try:
            self.predictor = NPKPredictor(
                unet_model_path="checkpoints/best_model.pth",
                regression_model_path="checkpoints/regression_model.pkl"
            )
            return "✅ Models loaded successfully!"
        except Exception as e:
            error_msg = f"❌ Error loading models: {str(e)}"
            print(error_msg)
            return error_msg
    
    def predict_npk_from_image(self, image, use_gpu=True):
        """
        Main prediction function for Gradio interface
        
        Args:
            image: PIL Image or numpy array from Gradio
            use_gpu: Whether to use GPU acceleration
        
        Returns:
            tuple: (results_text, visualization_image)
        """
        if self.predictor is None:
            return "❌ Models not loaded. Please check model files.", None
        
        if image is None:
            return "❌ Please upload an image first.", None
        
        try:
            # Save uploaded image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                if isinstance(image, np.ndarray):
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(image)
                else:
                    pil_image = image
                
                # Save as RGB
                pil_image.convert('RGB').save(tmp_file.name, 'JPEG')
                temp_path = tmp_file.name
            
            # Make prediction
            results = self.predictor.predict_from_image(temp_path, use_gpu=use_gpu)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if results is None:
                return "❌ Prediction failed. Please try again.", None
            
            # Format results
            results_text = self.format_results(results)
            
            # Create visualization
            viz_image = self.create_visualization(image, results)
            
            return results_text, viz_image
            
        except Exception as e:
            error_msg = f"❌ Error during prediction: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg, None
    
    def format_results(self, results):
        """Format prediction results for display"""
        if results is None:
            return "No results available"
        
        # Extract values
        cluster_areas = results['cluster_areas']
        approx_npk = results['approximate_npk']
        predicted_npk = results['predicted_npk']
        
        # Create formatted output
        output = []
        output.append("🔬 **NPK Prediction Results**")
        output.append("=" * 40)
        
        # Cluster areas
        output.append("\n📊 **Cluster Areas:**")
        cluster_names = ["White Beads", "Stain Beads", "Red Beads", "Black Beads"]
        for i, (name, area) in enumerate(zip(cluster_names, cluster_areas)):
            output.append(f"  • {name}: {area:,} pixels")
        
        total_area = sum(cluster_areas)
        output.append(f"  • **Total Area**: {total_area:,} pixels")
        
        # Percentages
        output.append("\n📈 **Composition Percentages:**")
        for i, (name, area) in enumerate(zip(cluster_names, cluster_areas)):
            percentage = (area / total_area * 100) if total_area > 0 else 0
            output.append(f"  • {name}: {percentage:.1f}%")
        
        # NPK Values
        output.append("\n🧪 **NPK Analysis:**")
        output.append("\n**Approximate NPK (Rule-based):**")
        output.append(f"  • Nitrogen (N): {approx_npk[0]:.2f}%")
        output.append(f"  • Phosphorus (P): {approx_npk[1]:.2f}%")
        output.append(f"  • Potassium (K): {approx_npk[2]:.2f}%")
        
        output.append("\n**Predicted NPK (ML Model):**")
        output.append(f"  • Nitrogen (N): {predicted_npk[0]:.2f}%")
        output.append(f"  • Phosphorus (P): {predicted_npk[1]:.2f}%")
        output.append(f"  • Potassium (K): {predicted_npk[2]:.2f}%")
        
        return "\n".join(output)
    
    def create_visualization(self, original_image, results):
        """Create a visualization of the prediction results"""
        try:
            # Convert to numpy array if needed
            if isinstance(original_image, Image.Image):
                img_array = np.array(original_image)
            else:
                img_array = original_image
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('NPK Prediction Analysis', fontsize=16, fontweight='bold')
            
            # Original image
            ax1.imshow(img_array)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Cluster areas bar chart
            cluster_names = ['White', 'Stain', 'Red', 'Black']
            cluster_areas = results['cluster_areas']
            colors = ['lightgray', 'orange', 'red', 'black']
            
            bars = ax2.bar(cluster_names, cluster_areas, color=colors, alpha=0.7)
            ax2.set_title('Cluster Areas (pixels)')
            ax2.set_ylabel('Area (pixels)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, area in zip(bars, cluster_areas):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{area:,}', ha='center', va='bottom', fontsize=8)
            
            # NPK comparison chart
            approx_npk = results['approximate_npk']
            predicted_npk = results['predicted_npk']
            
            x = np.arange(3)
            width = 0.35
            
            ax3.bar(x - width/2, approx_npk, width, label='Area approximated', alpha=0.7, color='lightblue')
            ax3.bar(x + width/2, predicted_npk, width, label='Regression predicted', alpha=0.7, color='darkblue')
            
            ax3.set_title('NPK Comparison')
            ax3.set_ylabel('Percentage (%)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(['N', 'P', 'K'])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (approx, pred) in enumerate(zip(approx_npk, predicted_npk)):
                ax3.text(i - width/2, approx + 0.5, f'{approx:.1f}', 
                        ha='center', va='bottom', fontsize=8)
                ax3.text(i + width/2, pred + 0.5, f'{pred:.1f}', 
                        ha='center', va='bottom', fontsize=8)
            
            # Composition pie chart
            total_area = sum(cluster_areas)
            if total_area > 0:
                percentages = [(area/total_area)*100 for area in cluster_areas]
                # Only show non-zero percentages
                non_zero_idx = [i for i, p in enumerate(percentages) if p > 0]
                if non_zero_idx:
                    pie_labels = [cluster_names[i] for i in non_zero_idx]
                    pie_values = [percentages[i] for i in non_zero_idx]
                    pie_colors = [colors[i] for i in non_zero_idx]
                    
                    wedges, texts, autotexts = ax4.pie(pie_values, labels=pie_labels, 
                                                      colors=pie_colors, autopct='%1.1f%%',
                                                      startangle=90)
                    ax4.set_title('Bead Composition')
                else:
                    ax4.text(0.5, 0.5, 'No beads detected', ha='center', va='center', 
                            transform=ax4.transAxes)
                    ax4.set_title('Bead Composition')
            else:
                ax4.text(0.5, 0.5, 'No area detected', ha='center', va='center', 
                        transform=ax4.transAxes)
                ax4.set_title('Bead Composition')
            
            plt.tight_layout()
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to PIL Image
            viz_image = Image.open(buf)
            plt.close(fig)
            
            return viz_image
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .output-text {
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }
        """
        
        # Create the interface
        with gr.Blocks(css=custom_css, title="NPK Soil Analysis") as interface:
            
            # Header
            gr.Markdown("""
            # 🌱 NPK Soil Analysis System
            
            Upload an image of soil beads to analyze NPK (Nitrogen, Phosphorus, Potassium) content.
            The system uses a U-Net model for bead segmentation and machine learning for NPK prediction.
            
            **Instructions:**
            1. Upload a clear image of soil beads
            2. Choose whether to use GPU acceleration (if available)
            3. Click "Analyze NPK" to get results
            """)
            
            # Model status
            with gr.Row():
                model_status = gr.Textbox(
                    value=self.initialize_predictor(),
                    label="Model Status",
                    interactive=False
                )
            
            # Main interface
            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    gr.Markdown("### 📤 Input")
                    
                    image_input = gr.Image(
                        label="Upload Soil Bead Image",
                        type="pil",
                        height=300
                    )
                    
                    use_gpu = gr.Checkbox(
                        label="Use GPU Acceleration",
                        value=True,
                        info="Enable if you have a CUDA-compatible GPU"
                    )
                    
                    analyze_btn = gr.Button(
                        "🔬 Analyze NPK",
                        variant="primary",
                        size="lg"
                    )
                    
                    # Example images section
                    gr.Markdown("### 📋 Example Images")
                    gr.Examples(
                        examples=[
                            # Add example image paths here if available
                            # ["examples/sample1.jpg"],
                            # ["examples/sample2.jpg"],
                        ],
                        inputs=[image_input],
                        label="Try these examples"
                    )
                
                with gr.Column(scale=2):
                    # Output section
                    gr.Markdown("### 📊 Results")
                    
                    with gr.Row():
                        with gr.Column():
                            results_text = gr.Textbox(
                                label="NPK Analysis Results",
                                value="Upload an image and click 'Analyze NPK' to see results here.",
                                lines=20,
                                max_lines=25,
                                elem_classes=["output-text"]
                            )
                        
                        with gr.Column():
                            visualization = gr.Image(
                                label="Analysis Visualization",
                                height=400
                            )
            
            # Advanced options (collapsed by default)
            with gr.Accordion("⚙️ Advanced Options", open=False):
                gr.Markdown("""
                **Model Information:**
                - U-Net model performs bead segmentation into 4 classes
                - Regression model predicts NPK values from bead areas
                - GPU acceleration requires CUDA-compatible hardware
                
                **Bead Types:**
                - White beads: High nitrogen content
                - Stain beads: Medium nitrogen content  
                - Red beads: High potassium content
                - Black beads: High phosphorus content
                """)
            
            # Set up the prediction function
            analyze_btn.click(
                fn=self.predict_npk_from_image,
                inputs=[image_input, use_gpu],
                outputs=[results_text, visualization],
                show_progress=True
            )
            
            # Footer
            gr.Markdown("""
            ---
            **Note:** This is a research tool. For agricultural decisions, please consult with soil testing professionals.
            """)
        
        return interface

def main():
    """Main function to run the Gradio app"""
    app = NPKGradioApp()
    interface = app.create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=True,            # Set to True to create a public link
        debug=True,             # Enable debug mode
        show_error=True,         # Show detailed error messages
        inbrowser=True  # Auto-open in browser
    )

if __name__ == "__main__":
    main()