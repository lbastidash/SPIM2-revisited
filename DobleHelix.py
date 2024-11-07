#FinalAlignementTest
import cv2
import numpy as np
from pycromanager import Bridge
import matplotlib.pyplot as plt
import aotools
import json
from UtilitySPIM2 import matriarch
from tqdm import tqdm
import slmpy
import slmAberrationCorrection
from slmAberrationCorrection import make_now
from slmAberrationCorrection import better_iterate
import logging
import time 
logging.basicConfig(level=logging.DEBUG)


logging.info("initializing")
#Opens a JSON file containing all the configurations needed
with open('config.json', 'r') as f:
    config = json.load(f)

# Y,X coordinates
slmShape = config["slm_device"]["resolution"]
fouriershape = config["fourier_properties"]["size"]
centerpoint = config["fourier_properties"]["center"]
stretch = config["fourier_properties"]["stretch"]
degree = config["settings"]["zernike_modes"]
epsilon = config["settings"]["iteration_epsilon"]
g_0 = config["settings"]["iteration_gain0"]
laser = config["illumination_device"]["name"]

guideStarSize, integralRadious = slmAberrationCorrection.make_now.calculate_guidestar_params(config["guide_star"]["microbead"], config["guide_star"]["binning"])

#HelixPhaseMask
ogMask = make_now.generate_corkscrew_optimized(int(fouriershape[0]/2))
angledMask = matriarch.stretch_image(ogMask, stretch)
display = np.zeros(slmShape)
DoubleHelixphaseMask = matriarch.frame_image(display, angledMask, centerpoint)

#Conecting to SLM and CORE
logging.info("Connecting to SLM")
slm = slmpy.SLMdisplay(monitor = config["slm_device"]["display"])
bridge = Bridge()
core = bridge.get_core()

while True:
    logging.info("Preparing Zero Mask")
    #Zero Mask Callibration
    phaseMask = np.zeros(slmShape)
    slm.updateArray(phaseMask.astype('uint8'))
    logging.info("Displaying phase mask")

    input("Press Enter to change to the DH Mask")

    #Adaptive Optics and Double Helix
    logging.info("Preparing DH Mask correction")
    completeMask = DoubleHelixphaseMask
    slm.updateArray(completeMask.astype('uint8'))
    logging.info("Displaying phase mask")

    input("Press Enter to change to the Zero Mask")

    endProcces=input("End process? Y/N")
    if endProcces != "N":
        break