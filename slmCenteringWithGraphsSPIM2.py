"""_summary_
SLM Centering Functions
Contains the Functions that run through the SLM at different centering coordinates.
By Artemis the Lynx, correspondence c.castelblancov@uniandes.edu.co 
Version 3.1 2024-10-01  
"""

#Dependencies
import aotools
from datetime import datetime
from auxiliarySPIM2 import slmpack
from auxiliarySPIM2 import zernike
import cv2
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import math
import os 
import pandas as pd
from pycromanager import Bridge
import slmpy
import scipy
from tqdm import tqdm
import UtilitySPIM2

#Logging Config
logging.basicConfig(level=logging.INFO)
logging.info("Initialization")

#Opens a JSON file containing all the configurations needed
with open('config.json', 'r') as f:
    config = json.load(f)
slm=config["slm_device"]["resolution"]

 
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
    percentile_99 = np.percentile(imagePlus, 99)
    imagePlus[imagePlus < percentile_99] = 0
    cmassPlus = center_of_mass(imagePlus)
    img_displayPlus = imagePlus.astype(np.uint16)

    """_minusDefocus_
    """
    slm.updateArray(mPlaced.astype('uint8'))
    imageMinus = takeImage()
    percentile_99 = np.percentile(imageMinus, 99)
    imageMinus[imageMinus < percentile_99] = 0
    cmassMinus = center_of_mass(imageMinus)
     
    """_CV real time plotting
    """
    img_display = np.maximum(imageMinus, imagePlus)
    img_display = cv2.normalize(img_display, None, 0, 255, cv2.NORM_MINMAX)
    img_display = np.uint8(img_display)
    img_display = cv2.applyColorMap(img_display, cv2.COLORMAP_BONE)
    img_display = cv2.line(img_display, (int(cmassMinus[1]),int(cmassMinus[0])), (int(cmassPlus[1]),int(cmassPlus[0])), (0, 255, 255), 2)
    cv2.imshow('Real time Centers of Mass', img_display)
    
    #exit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        exit()

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



# Define the parameters for the surface evaluation
x_range = (-1920/2, 1920/2) #X resolution of the slm divided by 2
y_range = (-1154/2, 1154/2) #Y resolution of the slm divided by 2
x_steps = 10
y_steps = 5
laser = config["illumination_device"]["name"]


#mask Setup
logging.info("Calculating Phase Masks")

Blur = aotools.phaseFromZernikes([0,0,0,1],1000)
Blur = cv2.normalize(
    Blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


logging.debug("Calculating pawsitive Phase Mask")
pMask = UtilitySPIM2.matriarch.frame_image(np.zeros(slm), Blur, (1154//2, 1920//2) )
logging.debug("Calculating negative Phase Mask")
mMask = UtilitySPIM2.matriarch.frame_image(np.zeros(slm), (Blur*(-1)), (1154//2, 1920//2))


#Pycromanager Setup
logging.info("Accessing the Microscope")
slm = slmpy.SLMdisplay(monitor = config["slm_device"]["display"])
bridge = Bridge()
core = bridge.get_core()


# Evaluate the function on the surface
cv2.namedWindow('Real time Centers of Mass', cv2.WINDOW_NORMAL)


logging.info("Evaluating the centers of mass")
core.set_auto_shutter(False)
core.set_shutter_open(laser ,True)
surface_values = evaluate_defocus_on_surface(center_of_mass_difference, x_range, y_range, x_steps, y_steps, pMask, mMask)
ValuesXY = np.transpose(surface_values)
core.set_auto_shutter(False)
core.set_shutter_open(laser ,False)

#Closes the slm for safety
logging.info("closing slm connection")
slm.close()


# Plots the heatmap
logging.info("plotting results")
plt.imshow(ValuesXY, cmap='YlGn', extent=[x_range[0], x_range[1],  y_range[1], y_range[0]])
plt.colorbar(label='Radious')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Radius between center of masses')


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