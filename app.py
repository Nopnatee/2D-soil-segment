import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from script_4_app import (img_path_list, 
                     load_img_as_rgb, 
                     enhance_image, 
                     get_all_grouped_points, 
                     get_cluster_mask, 
                     get_area, 
                     get_npk)


# Enhanced wrapper function to get NPK and generate images
def get_npk_and_images(folder, number, npk_grid_data, shadow_area):
    """Enhanced wrapper function to get NPK from image and generate visualization images"""
    try:
        img_path = img_path_list[folder][number]
        w_comp = {'N': npk_grid_data.iloc[0, 1], 'P': npk_grid_data.iloc[0, 2], 'K': npk_grid_data.iloc[0, 3]}
        r_comp = {'N': npk_grid_data.iloc[1, 1], 'P': npk_grid_data.iloc[1, 2], 'K': npk_grid_data.iloc[1, 3]}
        s_comp = {'N': npk_grid_data.iloc[2, 1], 'P': npk_grid_data.iloc[2, 2], 'K': npk_grid_data.iloc[2, 3]}
        b_comp = {'N': npk_grid_data.iloc[3, 1], 'P': npk_grid_data.iloc[3, 2], 'K': npk_grid_data.iloc[3, 3]}

        # Get NPK result
        npk_result = get_npk(img_path, w_comp, r_comp, s_comp, b_comp, shadow_area)
        
        # Generate initial image
        initial_image = load_img_as_rgb(img_path)
        
        return npk_result, initial_image
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        # Return error message and None images
        return error_msg, None, None

# Create Gradio Interface
def create_app():
    """Create the main Gradio application"""
    
    with gr.Blocks(title="Soil 2D Segmentation Analysis Approximating N-P-K Composition", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
# Soil 2D Segmentation Analysis Approximating N-P-K Composition
- This script analyzes images of fertilizer beads to estimate their NPK (Nitrogen-Phosphorus-Potassium) composition using image enhancement, clustering, and pixel area analysis.
## 🧠 How it works:
### Image Loading & Enhancement:
- Loads all .jpg/.png images from specific folders.
- Enhances contrast, saturation, and brightness while preserving pure green pixels (used as background or markers).
### Clustering with KMeans:
- Non-green pixels from each image are clustered into 4 groups using KMeans based on color similarity.
### Mask Creation & Sorting:
- Generates a binary mask for each cluster.
- Sorts clusters by average brightness (from brightest to darkest), assuming this brightness correlates with specific bead types (white, stain, red, black).
### Area Calculation:
- Measures the pixel area of each cluster to estimate the quantity of each bead type.
### NPK Estimation:
- Applies predefined nutrient compositions to each bead color.
- Calculates the total and average NPK composition based on the bead areas.
### Main Execution:
- Tries to process the image print its NPK composition.
        """)

        with gr.Tabs():
            
            # Tab 1: Chapter 1 - Basic Communication
            with gr.TabItem("Load Image From Computer"):
                gr.Markdown("""This will load image from folder |pictures/14-7-35|, |pictures/15-7-18|, |pictures/15-15-15|, |pictures/18-4-5| from you current directory.""")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Select Image**")
                        folder = gr.Number(label="Folder Index", value=0, precision=0, 
                                      info="Select the folder index (0-3) containing images: 0 for 14-7-35, 1 for 15-7-18, 2 for 15-15-15, 3 for 18-4-5.")
                        number = gr.Number(label="Image Index", value=0, precision=0,
                                      info="Select the image number within the folder (0-?).")
                        gr.Markdown("**Bead NPK Compositions (N, P, K)**")
                        npk_grid_data = gr.Dataframe(
                            label="NPK Composition",
                            headers=["Color", "N", "P", "K"],
                            datatype=["str","number", "number", "number"],
                            row_count=4,
                            col_count=4,
                            value=[
                                ["White", 46, 0, 0],   # White
                                ["Red", 0, 0, 60],   # Red
                                ["Stain", 21, 0, 0],   # Stain
                                ["Black", 18, 46, 0]   # Black
                            ],
                            interactive=True
                        )
                        gr.Markdown("**Shadow Area**")
                        shadow_area = gr.Number(label="Shadow Area", value=200000, precision=2,
                                              info="Area of the shadow in pixels, used to exclude it from NPK calculations.")
                        with gr.Row():
                            analyze_btn1 = gr.Button("🔍 Analyze N-P-K Composition", variant="primary")
                    
                    with gr.Column():
                        loading_status1 = gr.Markdown("", visible=False)
                        gr.Markdown("**Original Image**")
                        initial_image_output = gr.Image(
                            label="Initial Picture",
                            type="numpy",
                            interactive=False,
                            show_label=True
                        )
                        output1 = gr.Markdown(label="Analyzed N-P-K Composition", value="**🔍 Press 'Analyze N-P-K Composition' to start analyzing.**")
                
                # Click event with loading states and image outputs
                analyze_btn1.click(
                    fn=lambda: (gr.update(value="🔄 **Analyzing...**", visible=True), 
                               gr.update(interactive=False), 
                               "", None, None),
                    outputs=[loading_status1, analyze_btn1, output1, initial_image_output]
                ).then(
                    fn=get_npk_and_images,
                    inputs=[folder, number, npk_grid_data, shadow_area],
                    outputs=[output1, initial_image_output]
                ).then(
                    fn=lambda: (gr.update(visible=False), gr.update(interactive=True)),
                    outputs=[loading_status1, analyze_btn1]
                )
            
            # Tab 2: Chapter 2 - I-Messages and Emotion Reflection
            with gr.TabItem("Under Construction"):
                gr.Markdown("### Under Construction")
            
    return app


# Launch the application
gr_output_log = []

if __name__ == "__main__":
    app = create_app()
    app.launch(
        share=True,  # Set to True if you want to create a public link
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Default Gradio port
        debug=True
    )