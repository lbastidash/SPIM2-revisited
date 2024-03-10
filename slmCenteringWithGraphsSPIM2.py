"""_summary_
SLM Centering Functions
Contains the Functions that run through the SLM at different centering coordinates.
Version 1.0 Release, Winter 2024-01 
By Artemis the Lynx
"""

#Dependencies
import numpy as np
import matplotlib.pyplot as plt
from auxiliarySPIM2 import slmpack
from auxiliarySPIM2 import zernike
from pycromanager import Bridge
import slmpy
import scipy
import math
from tqdm import tqdm
import logging
from datetime import datetime
import os 
import pandas as pd


#Logging Config
logging.basicConfig(level=logging.INFO)
 
def evaluate_defocus_on_surface(func, x_range, y_range, x_steps, y_steps, pMask=0, mMask=0):
    """
    Evaluates both a pMask and a mMask on a surface defined by X and Y ranges at regular intervals.

    Parameters:
        func: Function to evaluate.
        x_range: Tuple (x_min, x_max) defining the range of X values.
        y_range: Tuple (y_min, y_max) defining the range of Y values.
        x_steps: Number of steps along the X axis.
        y_steps: Number of steps along the Y axis.
        pMask: The phase mask that will be used to evaluate
        mMask: The inverse of the Phase Mask

    Returns:
        A 2D numpy array containing the evaluated masks values.
    """
    
    #Matrix definitions
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_values = np.linspace(x_min, x_max, x_steps)
    y_values = np.linspace(y_min, y_max, y_steps)
    surface = np.zeros((x_steps, y_steps))

    #Matrix rundown
    logging.debug("Entering the matrix rundown loop")
    for i, x in tqdm(enumerate(x_values), total=len(x_values), desc='Running thru columns', unit='column' ):
        for j, y in tqdm(enumerate(y_values), total=len(y_values),  desc='Running thru rows', unit='row', leave=False ):
            surface[i, j] = func(x, y, pMask, mMask)

    return surface


def center_of_mass_difference(x, y, pMask, mMask):
    """
    Evaluates the center of mass difference between a system with a pMask and a mMask

    Parameters:
        x: X coordinate where the phase mask will be centered at 
        y: Y coordinate where the phase mask will be centered at 
        pMask: phase mask of positive intensity 
        mMask: phase mask of negative intensity


    Returns:
        an array containing the difference between the centers of mass for each pMask/mMask centering
    """    
    
    pPlaced = slmpack.shift_image(pMask, int(x), int(y))
    mPlaced = slmpack.shift_image(mMask, int(x), int(y))

    """_PlusDefocus_
    """
    slm.updateArray(pPlaced.astype('uint8'))
    imagePlus = takeImage()
    cmassPlus = center_of_mass(imagePlus)
    

    """_minusDefocus_
    """
    slm.updateArray(mPlaced.astype('uint8'))
    imageMinus = takeImage()
    cmassMinus = center_of_mass(imageMinus)
    
    """#Center of Mass Evaluation
    """
    x1, y1 = cmassPlus
    x2, y2 = cmassMinus
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def center_of_mass (image):
    """
    returns the coordinates of the center of mass of an array

    Parameters:
        image: the image we want to calculate the center of mass of

    Returns:
        Tuple containing the X, Y coordinates of the Center of mass
    """
    bg = np.percentile(image, 99)
    roi_binary = np.zeros(image.shape)
    roi_binary[image > bg] = 1
    return scipy.ndimage.measurements.center_of_mass(roi_binary)


def takeImage():
    """
    #uses pycromanager to take an image 

    Parameters:
        void

    Returns:
        A 2D numpy array containing the values from the image.
    """
    core.snap_image()
    tagged_image = core.get_tagged_image()
    imageH = tagged_image.tags['Height']
    imageW = tagged_image.tags['Width']
    image = tagged_image.pix.reshape((imageH,imageW))
    logging.debug("image taken")
    return image  


logging.info("Initialization")
# Define the parameters for the surface evaluation
x_range = (-1920/2, 1920/2) #X resolution of the slm divided by 2
y_range = (-1152/2, 1152/2) #Y resolution of the slm divided by 2
x_steps = 10
y_steps = 5


#mask Setup
logging.info("Calculating Phase Masks")
logging.debug("Calculating positive Phase Mask")
pMask = zernike.Defocus(signo=1)
logging.debug("Calculating negative Phase Mask")
mMask = zernike.Defocus(signo=-1)


#Pycromanager Setup
logging.info("Accessing the Microscope")
slm = slmpy.SLMdisplay(monitor = 1)
bridge = Bridge()
core = bridge.get_core()


# Evaluate the function on the surface
logging.info("Evaluating the centers of mass")
surface_values = evaluate_defocus_on_surface(center_of_mass_difference, x_range, y_range, x_steps, y_steps, pMask, mMask)
ValuesXY = np.transpose(surface_values)


#Closes the slm for safety
logging.info("closing slm connection")
slm.close()


# Plots the heatmap
logging.info("plotting results")
plt.imshow(ValuesXY, cmap='YlGn', extent=[x_range[0], x_range[1],  y_range[1], y_range[0]])
plt.colorbar(label='Radious')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Radious between center of masses.')


#saves the data
logging.info("Saving Data")

pandaValued = pd.DataFrame(ValuesXY)
current_datetime = datetime.now()
folder_name = f"{current_datetime.strftime('%Y-%m-%d')}_slmCentering"
os.makedirs(folder_name, exist_ok=True)

filename = f"{current_datetime.strftime('%Y-%m-%d')}_radiousPerCoordinate.csv"
file_path = os.path.join(folder_name, filename)
pandaValued.to_csv(file_path, index = False)

plotname = f"{current_datetime.strftime('%Y-%m-%d')}_radiousPerCoordinate.pdf"
plot_path = os.path.join(folder_name, plotname)
plt.savefig(plot_path)
plt.show()
logging.info(f"Data saved to {file_path}")