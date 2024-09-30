#FinalAlignementTest
import cv2
import numpy as np
from pycromanager import Bridge
import matplotlib.pyplot as plt
import aotools
from UtilitySPIM2 import matriarch
from tqdm import tqdm
import slmpy
import slmAberrationCorrection
from slmAberrationCorrection import make_now
from slmAberrationCorrection import iterate
import logging
import time 
logging.basicConfig(level=logging.DEBUG)





# Y,X coordinates
slmShape = (1154,1920)
fouriershape = (1000,1000)
centerpoint = (660,860)
stretch = 1.01
degree, g_0, epsilon, totalIterations = 21, 2.5, 0.6, 30

#HelixPhaseMask
ogMask = make_now.generate_corkscrew_optimized(int(fouriershape[0]/2))
#angledMask = matriarch.stretch_image(ogMask, stretch)
display = np.zeros(slmShape)
#DoubleHelixphaseMask = matriarch.frame_image(display, angledMask, centerpoint)

#Adaptive Optics
logging.info("Connecting to SLM")
slm = slmpy.SLMdisplay(monitor = 1)
bridge = Bridge()
core = bridge.get_core()

#GuideStarCallibration
cv2.namedWindow('Real time guide star', cv2.WINDOW_NORMAL)
phaseMask = np.zeros(slmShape)
slm.updateArray(phaseMask.astype('uint8'))
input("Press Enter to continue...")

print("sampling Noise")

noise_sample = slmAberrationCorrection.adaptiveOpt.sample_noise(core, 100)
print("Noise Sampled")



for i in tqdm(range(1000), desc='CapturingPreviewFrames', unit='Img'):
       
        guideStar = slmAberrationCorrection.adaptiveOpt.better_get_guidestar(core, noise_sample, (301,301))
        
        #Shows the GuideStar
        img_display = guideStar
        img_display = cv2.normalize(img_display, None, 0, 255, cv2.NORM_MINMAX)
        img_display = np.uint8(img_display)
        img_display = cv2.applyColorMap(img_display, cv2.COLORMAP_BONE)
        img_display = cv2.circle(img_display, (img_display.shape[0] // 2, img_display.shape[1] // 2), 75, (0, 255, 255), 2)
        cv2.imshow('Real time guide star', img_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

logging.info("Preparing Phase Mask correction")
CorrectionPhasemask, iterations, metrics = iterate(slmShape, fouriershape, centerpoint, stretch, degree, g_0, epsilon, totalIterations, slm, preview=True, integR = 60, guidS =(201,201))


#Adaptive Optics and Double Helix
logging.info("Preparing Phase Mask correction")
#completeMask = DoubleHelixphaseMask + CorrectionPhasemask
#completeMask = completeMask - completeMask.min()
#completeMask = completeMask%255
#slm.updateArray(completeMask.astype('uint8'))

logging.info("Displaying phase mask")
#plt.imshow(completeMask)
#plt.show()

