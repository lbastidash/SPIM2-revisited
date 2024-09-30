"""_summary_
Finds the distribution of the Metric and uses an iterative process to obtain an optimal
value for the Epsilon constant used in Stochastic gradient descend
By Artemis the Lynx, correspondence c.castelblancov@uniandes.edu.co 
version 3.0 2024-09-30
"""
import numpy as np
import matplotlib.pyplot as plt
import aotools
import cv2
import json
from UtilitySPIM2 import matriarch
import slmpy
from slmAberrationCorrection import make_now
from slmAberrationCorrection import iterate
from pycromanager import Bridge, Acquisition
import logging
import msvcrt
from tqdm import tqdm
logging.basicConfig(level=logging.DEBUG)
from slmAberrationCorrection import adaptiveOpt

#Opens a JSON file containing all the configurations needed
with open('config.json', 'r') as f:
    config = json.load(f)

distroSamples = config["settings"]["metric_samples"]
metricTolerance = 1.1
degree = config["settings"]["zernike_modes"]
stepSize = 0.1
slmShape = config["slm_device"]["resolution"]
fouriershape = config["fourier_properties"]["size"]
centerpoint = config["fourier_properties"]["center"]
stretch = config["fourier_properties"]["stretch"]
laser = config["illumination_device"]["name"]

#Connect to the SLM
logging.info("Connecting to Microscope")
bridge = Bridge()
core = bridge.get_core()
slm = slmpy.SLMdisplay(monitor = config["slm_device"]["resolution"])
display = np.zeros(slmShape)
slm.updateArray(display.astype('uint8'))

print("Press Any key to confirm Acquisition")
msvcrt.getch()
logging.info("Sampling Images")
metrics = [] 

"""_Distribution Sampling Loop
"""
noiseSample = adaptiveOpt.sample_noise(core, laser, 100)

core.snap_image()#We snap a random image to know the size of the sensor
tagged_image = core.get_tagged_image()
SensorSize = (tagged_image.tags['Height'],tagged_image.tags['Width'])

core.start_sequence_acquisition(distroSamples, 0, False)
for i in tqdm(range(distroSamples), desc="Sampling", unit='sample'):#We start a continous acquisition mode that snaps frames quickly
    if core.get_remaining_image_count() > 0:
        img = core.get_last_image()
        image_i = adaptiveOpt.tf_into_guidestar(img, SensorSize, noiseSample, size=(201,201),  graph=False)
        m_i = adaptiveOpt.metric_better_r(image_i)
        metrics.append(m_i)
core.stop_sequence_acquisition()

"""_STD Calculation    
"""
logging.info("Calculating Distribution") 
Metrics = np.array(metrics)
average = np.mean(Metrics)
sigma = np.std(Metrics)

"""_STD Visualization    
"""
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

logging.info("Iterations finished mrrowr~") 
optimalValue = epsilon-stepSize
print(f"optimal Epsilon value Found E={optimalValue}")

update = input(f"Update the value in the config file to {optimalValue}? (yes/no): ")
if update == 'yes':
    # Update a variable in the JSON file (modify this as per your needs)
    config["settings"]["iteration_epsilon"] = optimalValue
    with open('config.json', 'w') as file:
        json.dump(config, file, indent=4)
        print("Configuration updated :3")
else:
    print("Configuration not updated.")
