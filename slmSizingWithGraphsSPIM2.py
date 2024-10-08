"""_summary_
SLM Sizing Functions
Contains the Functions that run through the SLM to find the optimal size
By Artemis the Lynx, correspondence c.castelblancov@uniandes.edu.co 
Version 3.2 2024-10-01
"""

import slmAberrationCorrection
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import msvcrt
import numpy as np
import pandas as pd
from pycromanager import Bridge
import slmpy
from slmAberrationCorrection import make_now
from tqdm import tqdm
import cv2
import os
from PIL import Image
import seaborn

#TODO Data acquisition disabled, running on live mode only

#Logging Config
logging.basicConfig(level=logging.INFO)
logging.info("Initialization")

#Opens a JSON file containing all the configurations needed
with open('config.json', 'r') as f:
    config = json.load(f)

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

def curtain(slmShape, cutout, value=128, axes = 1):
    matrix = np.zeros(slmShape)
    if axes == 1:
        matrix[:, :cutout] = value
        return matrix
    elif axes == 0:
        matrix[cutout:, :] = value  # Fill from start_row onwards with ones
        return matrix
    
def evaluate_binary(slmShape, coverRange, steps, core, laser):
    """
    Acquires average and maximum pixel intensity data after applying a binary curtain phase mask
    Parameters:
        slmShape: Tuple (x, y) resolution of the SLM 
        coverRange: Tuple (y_min, y_max) range the curtain mask will extend
        steps: int Number of steps
        core: pycromanager microscope core
        laser: 
    Returns:
        means, maxes, interval : lists with the mean and maximum pixel intensity and the curtain interval
    """
    
    means = []
    maxes = []
    metrics = []
    interval = np.linspace(coverRange[0], coverRange[1], steps)
    
    #NoiseSample
    noise_sample = slmAberrationCorrection.adaptiveOpt.sample_noise(core, laser)

    logging.info("Entering the simulation Loop")
    core.set_auto_shutter(False)
    core.set_shutter_open(laser ,True)
    for i in tqdm(range(len(interval)), desc='Running thru the SLM', unit='Img'):
        phaseMask = curtain(slmShape, int(interval[i]))
        slm.updateArray(phaseMask.astype('uint8'))
        guideStar = slmAberrationCorrection.adaptiveOpt.better_get_guidestar(core, noise_sample, starSize)
        
        #Shows the GuideStar
        img_display = guideStar
        img_display = cv2.normalize(img_display, None, 0, 255, cv2.NORM_MINMAX)
        img_display = np.uint8(img_display)
        img_display = cv2.applyColorMap(img_display, cv2.COLORMAP_BONE)
        img_display = cv2.circle(img_display, (img_display.shape[0] // 2, img_display.shape[1] // 2), metricRads, (0, 255, 255), 2)
        cv2.imshow('Real time guide star', img_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        means.append(np.mean(guideStar))
        maxes.append(guideStar.max())
        metrics.append(slmAberrationCorrection.adaptiveOpt.metric_better_r(guideStar,metricRads))
    core.set_shutter_open(laser ,False)
        

    logging.info("Simulation Loop Sucessful") 
    
    return means, maxes, interval, metrics

#Initializing    
SLMresolution = config["slm_device"]["resolution"]
coverRange = (10, 1900)
steps = 100
repetitions = 5
starSize, metricRads = make_now.calculate_guidestar_params(config["guide_star"]["microbead"], config["guide_star"]["binning"])
laser = config["illumination_device"]["name"]


#Pycromanager Setup
logging.info("Accessing the Microscope")
slm = slmpy.SLMdisplay(monitor = config["slm_device"]["display"])
bridge = Bridge()
core = bridge.get_core()

#Data Acquisition
logging.info("Starting data acquisition")
cv2.namedWindow('Real time guide star', cv2.WINDOW_NORMAL)
meanIntensity, maxIntensity, curtainRange, metrics = evaluate_binary(SLMresolution, coverRange, steps, core, laser)
Inertia = np.asarray(metrics)
for i in range(repetitions):
    print("Press Any key to continue Acquisition")
    msvcrt.getch()
    _, _, _, _, repetitionMetric = evaluate_binary(SLMresolution, coverRange, steps, core, laser)
    repetitionInertia = np.asarray(repetitionMetric)
    Inertia = Inertia + repetitionInertia

Inertia= Inertia/(repetitions+1)

logging.info("Data acquisition successful")
cv2.destroyAllWindows()

#closes the slm for safety
logging.info("closing slm connection")
#slm.close()

#Plot the Mean and metric values of the PSF    
logging.info("Plotting Results")
plt.figure(figsize=(10, 5))
# Plot metrics
plt.subplot(1, 2, 1)
plt.plot(curtainRange, Inertia, color='#240046', linewidth=2, linestyle='-')
plt.scatter(curtainRange, Inertia, marker='d', color='#B40424')
plt.title('PSF Metric',fontsize=20)
plt.grid(True, color='#E1E2EF')  # Add grid lines
plt.xlabel('Fourier plane occlusion', fontsize=18)
plt.ylabel('PSF Metric', fontsize=18)
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