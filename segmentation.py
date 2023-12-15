import numpy as np
import os 
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, color, filters, morphology, measure, img_as_ubyte

input_dir = Path(r"C:\Users\Laszlo\Desktop\fiji\medusa_ex\to analyze")
output_dir = Path(r"C:\Users\Laszlo\Desktop\fiji\medusa_ex\output")

for entry in os.listdir(input_dir):
    # Load the image
    image = io.imread(input_dir/ entry)  # Replace with the actual path

    # Gaussian blur
    blurred = filters.gaussian(image, sigma=2)

    # Thresholding
    threshold = filters.threshold_otsu(blurred)
    binary_image = blurred <= threshold

    # Dilation and Erosion
    binary_image = morphology.binary_dilation(binary_image, morphology.disk(5))
    binary_image = morphology.binary_erosion(binary_image, morphology.disk(5))

    # Fill Holes
    binary_image = morphology.remove_small_objects(binary_image, min_size=150000)

    # Label and measure regions
    label_image = measure.label(binary_image)
    properties = measure.regionprops(label_image)
    print(len(properties))

    # Find the largest region
    largest_region = max(properties, key=lambda prop: prop.area)
    label = largest_region.label
    filtered_image = label_image
    colored_image = color.label2rgb(filtered_image, bg_color="white")
    perimiter = largest_region.perimeter
    
    # detect 

    # Convert the boolean image to uint8 using img_as_ubyte
    binary_image_uint8 = img_as_ubyte(colored_image)

    # Get coordinates of the largest region
    io.imsave(output_dir / entry, binary_image_uint8)
