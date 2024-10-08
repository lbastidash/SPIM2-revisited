
"""
Utility Functions for SPIM2 related code 
Code that doesn't explicitly relate to the SLM or .CORE 
functions should go here. 
Artemis Castelblanco
Versión: 2.0 2024-09-30
"""


import numpy as np
import os
import cv2

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

    def stretch_image(image, stretch_factor, axis='x'):
            """
            Stretch an image array along a given axis by a stretch factor.
            
            Parameters:
                image (numpy.ndarray): Input image array.
                stretch_factor (float): Factor by which to stretch the image.
                axis (str): Axis along which to stretch the image ('x' or 'y').
            
            Returns:
                numpy.ndarray: Stretched image array.
            """
            if axis == 'x':
                new_width = int(image.shape[1] * stretch_factor)
                stretched_image = cv2.resize(image, (new_width, image.shape[0]))
            elif axis == 'y':
                new_height = int(image.shape[0] * stretch_factor)
                stretched_image = cv2.resize(image, (image.shape[1], new_height))
            else:
                raise ValueError("Invalid axis. Please use 'x' or 'y'.")
    
            return stretched_image
