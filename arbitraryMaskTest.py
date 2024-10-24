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
totalIterations = 100

guideStarSize, integralRadious = slmAberrationCorrection.make_now.calculate_guidestar_params(config["guide_star"]["microbead"], config["guide_star"]["binning"])

#HelixPhaseMask
ogMask = make_now.generate_corkscrew_optimized(int(fouriershape[0]/2))
angledMask = matriarch.stretch_image(ogMask, stretch)
display = np.zeros(slmShape)
DoubleHelixphaseMask = matriarch.frame_image(display, angledMask, centerpoint)

#Adaptive Optics
logging.info("Connecting to SLM")
slm = slmpy.SLMdisplay(monitor = config["slm_device"]["display"])
bridge = Bridge()
core = bridge.get_core()
logging.info("Please check guide Star visibility before proceeding")
#GuideStarCallibration
phaseMask = np.zeros(slmShape)
slm.updateArray(phaseMask.astype('uint8'))

input("Press Enter to continue...")

print("sampling Noise")
noise_sample = slmAberrationCorrection.adaptiveOpt.sample_noise(core, laser, 50)
print("Noise Sampled")
logging.info("Preparing Phase Mask correction")
CorrectionPhasemask, iterations, metrics = better_iterate(noise_sample, totalIterations, slm)
input("Press Enter to continue...")
plt.scatter(iterations, metrics)
plt.show()

#Adaptive Optics and Double Helix
logging.info("Preparing Phase Mask correction")
completeMask = DoubleHelixphaseMask + CorrectionPhasemask
completeMask = completeMask - completeMask.min()
completeMask = completeMask%256
slm.updateArray(completeMask.astype('uint8'))

logging.info("Displaying phase mask")
plt.imshow(completeMask)
plt.show()

