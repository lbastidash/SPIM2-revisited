"""_summary_
SLM Sizing Functions
Contains the Functions that run through the SLM to find the optimal size
Version 1.20240414 Release, Winter 2024-01 
By Artemis the Lynx
"""
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycromanager import Bridge
import slmpy
from tqdm import tqdm
from auxiliarySPIM2 import lumiere
import os

#Logging Config
logging.basicConfig(level=logging.INFO)

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

def curtain(slmShape, cutout, value=1):
    matrix = np.zeros(slmShape)
    matrix[:, :cutout] = value
    return matrix

def evaluate_binary(slmShape, coverRange, steps, core):
    """
    Acquires average and maximum pixel intensity data after applying a binary curtain phase mask
    Parameters:
        slmShape: Tuple (x_min, x_max) resolution of the SLM 
        coverRange: Tuple (y_min, y_max) range the curtain mask will extend
        steps: Number of steps
        core: pycromanager microscope core
    Returns:
        means, maxes, interval : lists with the mean and maximum pixel intensity and the curtain interval
    """
    
    means = []
    maxes = []
    interval = np.linspace(coverRange[0], coverRange[1], steps)
    
    logging.info("Entering the simulation Loop")
    for i in tqdm(range(len(interval)), desc='Running thru the SLM', unit='Img'):
        phaseMask = curtain(slmShape, int(interval[i]))
        slm.updateArray(phaseMask.astype('uint8'))
        #microData = takeImage()
        microData = lumiere.take_Image(core)
        means.append(np.mean(microData))
        maxes.append(microData.max())
    logging.info("Simulation Loop Sucessful") 
    
    return means, maxes, interval

#Initializing    
logging.info("Initialization")
SLMresolution = (1920, 1034)
coverRange = (800, 900)
steps = 100

#Pycromanager Setup
logging.info("Accessing the Microscope")
slm = slmpy.SLMdisplay(monitor = 1)
bridge = Bridge()
core = bridge.get_core()

#Data Acquisition
logging.info("Starting data acquisition")
meanIntensity, maxIntensity, curtainRange = evaluate_binary(SLMresolution, coverRange, steps, core)
logging.info("Data acquisition successful")

#closes the slm for safety
logging.info("closing slm connection")
slm.close()

#Plot the Mean and Maximum values of the PSF    
logging.info("Plotting Results")
plt.figure(figsize=(10, 5))
# Plot maximums
plt.subplot(1, 2, 1)
plt.plot(curtainRange, maxIntensity, color='#240046', linewidth=2, linestyle='-')
plt.scatter(curtainRange, maxIntensity, marker='d', color='#B40424')
plt.title('PSF Maximum Intensities',fontsize=20)
plt.grid(True, color='#E1E2EF')  # Add grid lines
plt.xlabel('Fourier plane occlusion', fontsize=18)
plt.ylabel('Maximum PSF intensity', fontsize=18)
# Plot averages
plt.subplot(1, 2, 2)
plt.plot(curtainRange, meanIntensity, color='#240046', linewidth=2, linestyle='-')
plt.scatter(curtainRange, meanIntensity,  marker='d', color='#B40424')
plt.title('PSF Mean Intensity', fontsize=20)
plt.grid(True, color='#E1E2EF')  # Add grid lines
plt.xlabel('Fourier plane occlusion', fontsize=18)
plt.ylabel('Mean PSF Intesity', fontsize=18)

#saves the data
logging.info("Saving Data")
#saves the maximums and averages using pandas 
df = pd.DataFrame({'Range': curtainRange, 'Means': meanIntensity, 'Maximums': maxIntensity})
current_datetime = datetime.now()
folder_name = f"{current_datetime.strftime('%Y-%m-%d')}_slmSizing"
os.makedirs(folder_name, exist_ok=True)
filename = f"{current_datetime.strftime('%Y-%m-%d')}_slmSizingData.csv"
file_path = os.path.join(folder_name, filename)
df.to_csv(file_path, index = False)
#saving plot
plotname = f"{current_datetime.strftime('%Y-%m-%d')}_intensitiesAndAverages.pdf"
plot_path = os.path.join(folder_name, plotname)
plt.savefig(plot_path)
plt.show()
#program done
logging.info(f"Data saved to {file_path}")
print(':3')