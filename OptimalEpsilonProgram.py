#Optimal Epsilon
import numpy as np
import matplotlib.pyplot as plt
import aotools
import cv2
from UtilitySPIM2 import matriarch
import slmpy
from slmAberrationCorrection import make_now
from slmAberrationCorrection import iterate
from pycromanager import Bridge
import logging
import msvcrt
from tqdm import tqdm
logging.basicConfig(level=logging.DEBUG)
from slmAberrationCorrection import adaptiveOpt

distroSamples = 100
metricTolerance = 1.1
degree = 21
stepSize = 0.1

# Y,X coordinates
slmShape = (1154,1920)
fouriershape = (1000,1000)
centerpoint = (660,860)
stretch = 1.01

#Connect to the SLM
logging.info("Connecting to Microscope")
bridge = Bridge()
core = bridge.get_core()
slm = slmpy.SLMdisplay(monitor = 1)
display = np.zeros(slmShape)
slm.updateArray(display.astype('uint8'))

print("Press Any key to confirm Depth")
msvcrt.getch()
print("Depth Confirmed")

logging.info("Taking Images")
metrics = [] 

"""_Distribution Sampling Loop
"""
for i in tqdm(range(distroSamples), desc="Sampling", unit='sample'):
    image_i = adaptiveOpt.get_guidestar(core,graph=False)
    m_i = adaptiveOpt.metric_better_r(image_i)
    metrics.append(m_i)


"""_STD Calculation    
"""
logging.info("Calculating Distribution") 
Metrics = np.array(metrics)
average = np.mean(Metrics)
sigma = np.std(Metrics)

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(Metrics, bins=10, alpha=0.6, color='g', edgecolor='black')
# Highlight the average and standard deviation
plt.axvline(average, color='r', linestyle='dashed', linewidth=2, label=f'Average: {average:.2f}')
plt.axvline(average - sigma, color='b', linestyle='dashed', linewidth=2, label=f'-1 Std Dev: {average - sigma:.2f}')
plt.axvline(average + sigma, color='b', linestyle='dashed', linewidth=2, label=f'+1 Std Dev: {average + sigma:.2f}')
# Add labels and title
plt.title('Metric Distribution for an unchanging image')
plt.xlabel('Metric')
plt.ylabel('Frequency')
# Add legend
plt.legend()
# Show plot
plt.show()


#Optimal Epsilon Iteration 
diff = 0

Zernikes= aotools.zernike.zernikeArray(degree,fouriershape[0])
N = len(Zernikes)
array =  (int(N)*[0])
a_t = np.array(array)
C_t= (make_now.random_signs(len(a_t)))
epsilon = stepSize

criteria = sigma*metricTolerance
logging.info("Entering the Iteration Cycle") 

cv2.namedWindow('Binarized difference', cv2.WINDOW_NORMAL)

with tqdm(total=criteria) as pbar:
    while diff < criteria:
        
        D_t = epsilon*C_t
        a_plus = a_t + D_t
        a_minus = a_t - D_t
        
        #Creates phase masks 
        phaseMask_p = make_now.zernike_optimized(Zernikes, a_plus, stretch, slmShape, centerpoint)
        phaseMask_m = make_now.zernike_optimized(Zernikes, a_minus, stretch, slmShape, centerpoint)

        #takes phase masked images
        slm.updateArray(phaseMask_p.astype('uint8'))
        guideStar_p = adaptiveOpt.get_guidestar(core)
        slm.updateArray(phaseMask_m.astype('uint8'))
        guideStar_m = adaptiveOpt.get_guidestar(core)
            
        #Evaluates the metric for each phase mask image
        metric_p = adaptiveOpt.metric_better_r(guideStar_p)
        metric_m = adaptiveOpt.metric_better_r(guideStar_m)
        diff = metric_p - metric_m
        logging.debug(f"Metric Difference ={diff}")
        
        #plots the difference for comfort
        _, binary_pimage = cv2.threshold(guideStar_p, np.percentile(guideStar_p,99), 1, cv2.THRESH_BINARY)
        _, binary_mimage = cv2.threshold(guideStar_m, np.percentile(guideStar_m,99), 1, cv2.THRESH_BINARY)
        xorImage = (binary_mimage+binary_pimage)%2
        xorImage = cv2.applyColorMap(xorImage, cv2.COLORMAP_BONE)
        cv2.putText(xorImage, f"Epsilon = {epsilon}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('Binarized difference', xorImage)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()

        epsilon = epsilon + stepSize

        pbar.update(diff - pbar.n)

logging.info("Iterations finished") 
print(f"optimal Epsilon value Found E={epsilon-stepSize}")

    