import cv2
import numpy as np
from rembg import remove
import gradio as gr

# Function to process the image and replace the background
def process_image(image, new_background):
    # Remove the background from the image
    image_no_bg = remove(np.array(image))

    # Resize new background to match the original image dimensions
    new_background = cv2.resize(np.array(new_background), (image_no_bg.shape[1], image_no_bg.shape[0]))

    # Convert the background-removed image to RGB if necessary
    if image_no_bg.shape[-1] == 4:  # If alpha channel exists
        rgb_image_no_bg = cv2.cvtColor(image_no_bg, cv2.COLOR_BGRA2BGR)
    else:
        rgb_image_no_bg = image_no_bg

    # Create a mask from the alpha channel (if applicable)
    if image_no_bg.shape[-1] == 4:
        mask = image_no_bg[:, :, 3]  # Get alpha channel as mask
        mask = cv2.merge([mask, mask, mask])  # Convert mask to 3 channels
        mask = mask / 255.0  # Normalize mask to [0, 1]
    else:
        mask = np.ones_like(rgb_image_no_bg)  # Default mask if no alpha channel

    # Perform blending of the image with the new background
    result = (rgb_image_no_bg * mask + new_background * (1 - mask)).astype(np.uint8)

    return result

# Gradio interface
def gradio_interface(image, background):
    # Process the uploaded image and background
    result = process_image(image, background)
    return result

# Set up Gradio interface with two image inputs and one image output
iface = gr.Interface(
    fn=gradio_interface, 
    inputs=[
        gr.Image(label="Upload Image with Car", type="pil"),
        gr.Image(label="Upload New Background Image", type="pil")
    ], 
    outputs="image",
    title="Car Background Replacement",
    description="Upload an image of a car and a new background. The system will remove the background from the car image and place the car on the new background."
)

# Launch Gradio interface with a shareable link
iface.launch(share=True)
