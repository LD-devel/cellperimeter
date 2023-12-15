import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, color, filters, morphology, measure, img_as_ubyte, draw, segmentation

input_dir = Path(r"C:\Users\Laszlo\Desktop\fiji\medusa_ex\to analyze")
output_dir = Path(r"C:\Users\Laszlo\Desktop\fiji\medusa_ex\output")

class Intersection:
    def __init__(self, label_image, label) -> None:
        self.sides = {'l','r','t','b'}
        self.max_vals = {side : (None, None) for side in self.sides}
        self.min_vals = {side : (None, None) for side in self.sides}
        self.label_image = label_image
        self.width = len(label_image[0])
        self.height = len(label_image)

        # results:
        self.perimeter_diff = 0
        self.line_points = []
        
        self.__Compute_points_for_label(label)
    
    def __is_corner(self, x, y) -> bool:
        if x in {0, self.width-1} and y in {0, self.height-1}:
            return True
        return False
    
    def __is_border(self, x, y) -> bool:
        if x in {0, self.width-1} or y in {0, self.height-1}:
            return True
        return False

    def __get_borders(self, x, y):
        borders = []
        if x == 0:
            borders.append('l')
        if x == self.width-1:
            borders.append('r')
        if y == 0:
            borders.append('t')
        if y == self.height-1:
            borders.append('b')
        return borders
    
    def __update(self, x, y):
        # Determine whether this is a frame coordinate
        if not self.__is_border(x, y):
            return
        borders = self.__get_borders(x, y)
        assert len(borders) > 0

        # update min and max val
        for border in borders:
            min_x, min_y = self.min_vals[border]
            if min_x is None or x < min_x or y < min_y:
                self.min_vals[border] = (x, y)
            max_x, max_y = self.max_vals[border]
            if max_x is None or x > max_x or y > max_y:
                self.max_vals[border] = (x, y)
    
    def __Compute_points_for_label(self, label) -> None:
        # Update all the intersection points
        for x in range(self.width):
            for y in range(self.height):
                if self.label_image[y][x] == label:
                    self.__update(x,y)

        # Compute the perimeter diff
        for side in self.sides:
            min_x, min_y = self.min_vals[side]
            if min_x is None:
                continue
            max_x, max_y = self.max_vals[side]
            # Note: This is a debatable edge case.
            if not (self.__is_corner(min_x, min_y) or self.__is_corner(max_x, max_y)):
                continue
            self.perimeter_diff += np.linalg.norm([min_x - max_x, min_y - max_y], ord=1)
            
            # Detect end points of intersection
            if not self.__is_corner(min_x, min_y):
                self.line_points.append((min_x, min_y))
            if not self.__is_corner(max_x, max_y):
                self.line_points.append((max_x, max_y))
            
        # TODO: catch edge case that it intersects exactly in the corner.
        assert len(self.line_points) == 2, "Cell does not uniquely intersect at two sides."
        
    def get_perimeter_diff(self) -> int:
        assert self.perimeter_diff > 0, "Whoops, perimeter diff should not be 0."
        return self.perimeter_diff
    
    def get_points(self) -> list:
        return self.line_points

output_rows = []
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

    # Find the largest region
    largest_region = max(properties, key=lambda prop: prop.area)
    label = largest_region.label
    colored_image = color.label2rgb(label_image, colors=[(0,0,0)], bg_color="white")

    # Create a binary mask for the chosen region
    region_mask = (label_image == largest_region.label)

    # Use mark_boundaries to color only the perimeter of the region
    colored_image = segmentation.mark_boundaries(colored_image, region_mask, color=(0, 0, 1), mode='thick')  # Set color to red
    
    # detect 
    intersec = Intersection(label_image, label)
    points = intersec.get_points()
    rr, cc = draw.line(points[0][1],points[0][0],points[1][1],points[1][0])
    colored_image[rr, cc] = (1,0,0)

    # Convert the boolean image to uint8 using img_as_ubyte
    binary_image_uint8 = img_as_ubyte(colored_image)

    # Get coordinates of the largest region
    io.imsave(output_dir / entry, binary_image_uint8)
    
    # Export stats
    perimeter = largest_region.perimeter
    perimeter_cleaned = perimeter - intersec.get_perimeter_diff()
    line_length = np.linalg.norm([points[0][0]-points[1][0],points[0][1]-points[1][1] ], ord=1)
    perimeter_normalized = perimeter_cleaned / line_length
    new_row = {
       'filename': entry,
       'perimeter': perimeter, 
       'perimeter cleaned': perimeter_cleaned, 
       'perimeter normalized': perimeter_normalized,
    }
    output_rows.append(new_row)

df = pd.DataFrame(output_rows)
df.to_csv(output_dir/'Results.csv')
