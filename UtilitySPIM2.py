########
"""
Utility Functions for SPIM2 related code 
Code that doesn't explicitly relate to the SLM or .CORE 
functions should go here. 
Artemis Castelblanco
Versión: 0.1.20240305
"""
#######

import numpy as np
import os

######
"""
Matrix Related Functions
"""
######    
class matriarch():
    
    def frame_image (frame, image, center_point):
        """
        places the values of the image matrix inside the frame matrix with 
        center_point as the center point.
        Stable only where frame can contain image

        Parameters:
            frame (numpy.ndarray): Larger matrix to be modified.
            image (numpy.ndarray): Smaller matrix whose values will be placed into the larger matrix.
            center_point (tuple): Coordinates (row, column) specifying the center point.

        Returns:
            numpy.ndarray: Modified frame matrix.
        """
        smaller_rows, smaller_cols = image.shape
        center_row, center_col = center_point
        start_row = max(center_row - smaller_rows // 2, 0)
        end_row = min(start_row + smaller_rows, frame.shape[0])
        start_col = max(center_col - smaller_cols // 2, 0)
        end_col = min(start_col + smaller_cols, frame.shape[1])

        frame[start_row:end_row, start_col:end_col] = image[:end_row-start_row, :end_col-start_col]

        return frame


######
"""
File management related Functions
"""
######  
class librarian: 
    def save_data():
        #TODO Make a function that saves the files in the standard format
        pass
    
    
    
    
    def save_graph(plot, filename):
        """
        Save a matplotlib plot as a PDF file in the parent folder of the Python file.
        
        Parameters:
            plot (matplotlib.pyplot plot): The plot to save.
            filename (str): The filename (without extension) to save the plot as.
        """
        parent_folder = os.path.dirname(os.path.abspath(__file__))
        
        
        pass
    