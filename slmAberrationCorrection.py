"""_summary_
Aberration Correction: 
Contains functions that use Iterative process to correct for abberations
By Artemis the Lynx, correspondence c.castelblancov@uniandes.edu.co 
Version 4.1 2024-09-24
"""
import aotools
import cv2
from datetime import datetime
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import slmpy
import time
from UtilitySPIM2 import matriarch
from pycromanager import Bridge
from tqdm import tqdm
import msvcrt
import seaborn

logging.basicConfig(level=logging.DEBUG)

class make_now:
    
    def generate_corkscrew_optimized(radius,  N=7, center=(0, 0), fact=(3/4)): 
        x = np.linspace(-radius, radius, radius*2)
        y = np.linspace(-radius, radius, radius*2)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        Phi = np.arctan2(Y,X)
        
        answer = ((2*np.ceil((N)*((R/radius)**(1/fact)))-1)*Phi)%(2*np.pi) 
        
        return (answer)*aotools.circle(radius,2*radius)*(255/(2*np.pi))


    def random_signs(length):
        """generates a vector with random signs (-1) or (1)
        Args:
            length (int): lenght of the vector
        Returns:
            ndArray: vector filled with 1 and -1
        """
        array = np.zeros(length, dtype=int)  # Create an array of zeros with the specified length
        for i in range(length): 
            array[i] = random.choice([-1, 1])  # Set each element to either -1 or 1 randomly
        return array
    
    def zernike_optimized (Z_v, V_coef, strt, S_sz, C_cntr, norma=True, preview = False, maxAdmisible = 255, safeguard = 'fresnel'):
        """Optimized Zernike Polinomial phase mask makes

        Args:
            Z_v (ndArray): Array containing the Zernike Series
            V_coef (ndArray): Array with the coefficients
            strt (float): Stretch Factor
            S_sz (tuple): Size of the SLM
            C_cntr (tuple): center of the SLM
            norma (bool, optional): Wheter to normalize the phase mask or not. Defaults to True.
            preview (bool, optional): wheter to preview the phase mask or not. Defaults to False.
            maxAdmisible (int, optional): maximum value that the SLM can display. Defaults to 255.
            safeguard (str, optional): type of safeguard for if the value is surpassed. Defaults to 'truncating'.

        Returns:
            _type_: _description_
        """

        """
        for i in range(len(Z_v)):
            Z_v[i]=Z_v[i]*V_coef[i]
        """

        Z_i = np.zeros((Z_v.shape[1],Z_v.shape[2]))
        for i in range(len(V_coef)):
            Z_i = Z_i + Z_v[i]*V_coef[i]      
             
        
        Z_strt = matriarch.stretch_image(Z_i, strt)
        C_cnvas = np.zeros(S_sz)
        Z_result = matriarch.frame_image(C_cnvas, Z_strt, C_cntr )

        if norma: 
            Z_result = Z_result - Z_result.min()

        logging.debug(f"Minimum after recentering ={Z_result.min()}")
        


        if Z_result.max() > maxAdmisible:
            logging.warning(f"Maximum value {Z_result.max()} stretches over the admissible limit {maxAdmisible}")
            if safeguard == 'resize': 
                logging.warning(f"Rezising the Matrix")
                Z_result = Z_result*(maxAdmisible/Z_result.max())
            elif safeguard == 'fresnel': 
                logging.warning(f"Fresnel lens created with modulus {maxAdmisible}")
                Z_result = Z_result%maxAdmisible
            else: 
                logging.warning(f"Truncating the Matrix to {maxAdmisible}")
                Z_result = matriarch.truncate(Z_result, maxAdmisible)
        
        logging.debug(f"Maximum after resizing ={Z_result.max()}")

        if preview: 
            plt.imshow(Z_result)
            plt.show()
        
        return Z_result
    
    def zernike_phase_mask (V_coef, dmtr, strt, S_sz, C_cntr, norma=True, preview = True, maxAdmisible = 255, safeguard = 'fresnel'):
        Z_i = aotools.functions.zernike.phaseFromZernikes(V_coef, dmtr)
        Z_strt = matriarch.stretch_image(Z_i, strt)
        C_cnvas = np.zeros(S_sz)
        Z_result = matriarch.frame_image(C_cnvas, Z_strt, C_cntr )
        
        if norma: 
            Z_result = Z_result - Z_result.min()
        
        if Z_result.max() > maxAdmisible:
            logging.warning(f"Maximum value {Z_result.max()} stretches over the admissible limit {maxAdmisible}")
            if safeguard == 'resize': 
                logging.warning(f"Rezising the Matrix")
                Z_result = Z_result*(maxAdmisible/Z_result.max())
            elif safeguard == 'fresnel': 
                Z_result = Z_result%255
                logging.warning(f"Fresnel Lens yet to be implemented, Matrix left as it is")
            else: 
                logging.warning(f"Truncating the Matrix to {maxAdmisible}")
                Z_result = matriarch.truncate(Z_result, maxAdmisible)
        
        if preview: 
            plt.imshow(Z_result)
            plt.show()
        
        return Z_result

class adaptiveOpt:
    
    def metric_r_power_integral(img, integration_radius=60, power=2):
        """Metric of PSF quality based on integration of image(r) x r^2 over a circle of defined radius. 
        From Vorontsov, Shmalgausen, 1985 book. For best accuracy, img dimensions should be odd, with peak at the center.
        Parameters:
            img (array):  a 2D image with PSF peak at the center
            integration_radius (int) = for the circle of integration, default 20.
            power (int) = the power radious
        returns: 
            float
        """
        img = (img-img.min())/(img.max()-img.min())
        h, w = img.shape[0], img.shape[1]
        if np.min(img.shape) < 2 * integration_radius:
            raise ValueError("Radius too large for image size")
        else:
            cmass = scipy.ndimage.measurements.center_of_mass(img)
            x_center, y_center = cmass[1], cmass[0]
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
            mask = (dist_from_center <= integration_radius).astype(int)
            metric = np.sum(img * mask * (dist_from_center ** power)) 
        return metric
    
    def metric_better_r(img, integration_radius=40, preview=False):
        """A better metric for the iterative process

        Args:
            img (ndArray): Image we want the metric from   
            integration_radius (int, optional): radious we'll take into account for the metric. Defaults to 30.

        Returns:
            float: integration resutl
        """
        cmass = scipy.ndimage.measurements.center_of_mass(img)
        h, w = img.shape[0], img.shape[1]
        x_center, y_center = cmass[1], cmass[0]
        y, x = np.ogrid[:h, :w]
        radious = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
        mask = (radious <= integration_radius).astype(int)
        
        if preview:
            plt.imshow((img*mask*radious), cmap='mako')
            plt.title("Metric Representation")
            plt.show()
        
        metric = np.sum(img*mask*radious*radious)/np.sum(img*mask)
        
        return metric
    
    def extract_centered_matrix(matrix, center_coord, size):
        """
        Extract a smaller matrix centered around a specific coordinate from a larger matrix.

        Parameters:
            matrix (numpy.ndarray): Input matrix.
            center_coord (tuple): Coordinates (row, column) of the center.
            size (tuple): Size of the extracted matrix (rows, columns).

        Returns:
            numpy.ndarray: Extracted smaller matrix.
        """
        # Unpack center coordinates
        center_col, center_row = center_coord
            
        # Unpack size of the extracted matrix
        rows, cols = size
            
        # Calculate starting and ending indices for the extraction
        start_row = max(0, center_row - rows // 2)
        end_row = min(matrix.shape[0], center_row + (rows - rows // 2))
        start_col = max(0, center_col - cols // 2)
        end_col = min(matrix.shape[1], center_col + (cols - cols // 2))
            
        # Extract the smaller matrix
        smaller_matrix = matrix[start_row:end_row, start_col:end_col]
            
        return smaller_matrix

    def threshold_matrix(matrix, percentile=99.5):
        """
        Thresholds the values in a matrix below a certain percentile to 0.

        Parameters:
            matrix (numpy.ndarray): Input matrix.
            percentile (float): Percentile value below which to threshold.

        Returns:
            numpy.ndarray: Thresholded matrix.
        """
        # Calculate the threshold value based on the percentile
        threshold_value = np.percentile(matrix, percentile)
            
        # Threshold the matrix
        thresholded_matrix = np.where(matrix < threshold_value, 0, matrix)
        maxi = int(thresholded_matrix.max())
        result = np.where(matrix < 0, 0, thresholded_matrix)
        return result*(100/maxi)
    
    def center_of_mass(image):
        """
        Calculate the center of mass of an image matrix.

        Parameters:
            image (numpy.ndarray): Input image matrix.

        Returns:
            tuple: Coordinates (row, column) of the center of mass.
        """
        # Create grid of coordinates
        rows, cols = np.indices(image.shape)
        
        # Calculate total intensity
        total_intensity = np.sum(image)
        
        # Calculate center of mass coordinates
        center_row = np.sum(rows * image) / total_intensity
        center_col = np.sum(cols * image) / total_intensity
        
        return int(center_row), int(center_col)
    
    def sample_noise(core, samples=100):
        """Samples the baseline Noise from the camera
        Args:
            core (pycromanagerCore): the pycromanager microscope's core object
            samples (int): the number of samples used to calculate baseline noise
        Returns:
            Numpy Matrix: Matrix image of the average noise
        """

        core.snap_image()
        tagged_image = core.get_tagged_image()
        imageH = tagged_image.tags['Height']
        imageW = tagged_image.tags['Width']
        image = np.zeros((imageH, imageW))
        
        print("Initiating noise sampling, please make sure Laser is disabled before proceeding")
        input("Press Enter to continue...")

        for i in tqdm(range(samples), desc="Sampling", unit='sample'):
            core.snap_image()
            tagged_image = core.get_tagged_image()
            imageH = tagged_image.tags['Height']
            imageW = tagged_image.tags['Width']
            image = image + tagged_image.pix.reshape((imageH,imageW))
        
        sample = image/samples
        
        print("Noise sampling finished, please return the laser to its original configuration before proceeding")
        input("Press Enter to continue...")

        return sample
    
    
    def better_get_guidestar(core, noiseSample, size=(101,101), preview=False, kernel = (5,5)):
        """A better Function to obtain more accurate GuideStar Readings, requieres a noise sample
        Args:
            core (pycromanagerCore): the pycromanager microscope's core object
            noiseSample (Array): an array containing the noise sample for the 
            size (tuple, optional): size of the wanted gudiestar image. Defaults to (101,101).
            graph (bool, optional): wheter or not to show a preview of the guidestar image. Defaults to False.

        Returns:
            Numpy Matrix: Matrix image of the guidestar
        """
        
        core.snap_image()
        tagged_image = core.get_tagged_image()
        imageH = tagged_image.tags['Height']
        imageW = tagged_image.tags['Width']
        image = tagged_image.pix.reshape((imageH,imageW))

        cleaned_Star = image-noiseSample
        cleaned_Star[cleaned_Star < 0] = 0
        
        gaussian_Filtered = cv2.GaussianBlur(cleaned_Star, kernel, 0)

        thresh_Star = adaptiveOpt.threshold_matrix(gaussian_Filtered, 99)
        center = scipy.ndimage.measurements.center_of_mass(thresh_Star)
        
        centerOfMass=(int(center[1]),int(center[0]))

        guideStar = adaptiveOpt.extract_centered_matrix(cleaned_Star, centerOfMass, size)
        if preview:
            plt.imshow(guideStar, cmap="mako") 
            plt.colorbar
            plt.show()
        
        return guideStar 
    
    def get_guidestar(core, size=(101,101), graph=False):
        """Takes a photo of the sample in the microscope and returns a small matrix containing
        the guide star. Function only usable when no more than 1 fluorescent sample is visible in the microscope
        Args:
            core (pycromanagerCore): the pycromanager microscope's core object
            size (tuple, optional): size of the wanted gudiestar image. Defaults to (101,101).
            graph (bool, optional): wheter or not to show a preview of the guidestar image. Defaults to False.

        Returns:
            Numpy Matrix: Matrix image of the guidestar
        """
        
        core.snap_image()
        tagged_image = core.get_tagged_image()
        imageH = tagged_image.tags['Height']
        imageW = tagged_image.tags['Width']
        image = tagged_image.pix.reshape((imageH,imageW))

        threshStar = adaptiveOpt.threshold_matrix(image)
        center = scipy.ndimage.measurements.center_of_mass(threshStar)
        #centerOfMass = adaptiveOpt.center_of_mass(threshStar)
        centerOfMass=(int(center[1]),int(center[0]))

        guideStar = adaptiveOpt.extract_centered_matrix(threshStar,centerOfMass,size)
        if graph:
            plt.imshow(guideStar, cmap="mako") 
            plt.colorbar
            plt.show()
        
        return guideStar 

def iterate(slmShape, fouriershape, centerpoint, stretch, degree, g_0, epsilon, totalIterations, slm, preview=True, integR = 60, guidS =(201,201)):
    """_summary_

    Args:
        slmShape (tuple): Resolution of the SLM (Y,X)
        fouriershape (tuple): Resolution of the phase mask
        centerpoint (tuple): position of the phase Mask in the SLM
        stretch (float): stretch factor of the phaseMask
        degree (int): number of polinomials in the series
        g_0 (float): initial value for the learning rate
        epsilon (float): perturnation value
        totalIterations (int): number of iterations
        slm (sml object): slm function 
        preview (bool, optional): wheter or not to preview the results. Defaults to True.

    Returns:
        ndArray, list, list: _description_
    """
    Zernikes= aotools.zernike.zernikeArray(degree,fouriershape[0])
    N = len(Zernikes)
    array =  (int(N)*[0])

    logging.info("Accessing Microscope")
    #slm = slmpy.SLMdisplay(monitor = 1)
    bridge = Bridge()
    core = bridge.get_core()

    #Iteration Generator
    logging.info("Preadquisition")

    #Confirm Microsphere alignement

    slm.updateArray(np.zeros(slmShape))
    print("Press Any key to confirm Depth")
    msvcrt.getch()
    print("Depth Confirmed")

    #initializes the optimization variables
    startimage = adaptiveOpt.get_guidestar(core, guidS,graph=preview)
    M0 = adaptiveOpt.metric_better_r(startimage, integR)
    metric_t = M0
    a_dash_t = np.array(array)  

    #saves some data
    metrics = []
    polynomialSeries = []

    #main Iteration Cycle
    logging.info("10 seconds for iteration")
    time.sleep(10)
    logging.info("Iteration Begins")
    iteration = 1 
    iterations = []

    for i in tqdm(range(totalIterations), desc="Optimizing", unit='iteration'): 
    
        logging.debug(f"Coefficients ={a_dash_t}")

        #Disturbs the polynomial series Terms
        C_t= (make_now.random_signs(len(a_dash_t)))
        D_t = epsilon*C_t
        a_plus = a_dash_t + D_t
        a_minus = a_dash_t - D_t

        #Creates phase masks 
        phaseMask_p = make_now.zernike_optimized(Zernikes, a_plus, stretch, slmShape, centerpoint)
        phaseMask_m = make_now.zernike_optimized(Zernikes, a_minus, stretch, slmShape, centerpoint)

        #takes phase masked images
        slm.updateArray(phaseMask_p.astype('uint8'))
        guideStar_p = adaptiveOpt.get_guidestar(core, guidS)
        slm.updateArray(phaseMask_m.astype('uint8'))
        guideStar_m = adaptiveOpt.get_guidestar(core, guidS)
        
        #Evaluates the metric for each phase mask image
        metric_p = adaptiveOpt.metric_better_r(guideStar_p, integR)
        metric_m = adaptiveOpt.metric_better_r(guideStar_m, integR)
        diff = metric_p - metric_m
        logging.debug(f"Metric Difference ={diff}")
        
        #Creates the iteration's Phase mask
        phaseMask_t = make_now.zernike_optimized(Zernikes, a_dash_t,stretch, slmShape, centerpoint)
        slm.updateArray(phaseMask_t.astype('uint8'))
        guideStar_t = adaptiveOpt.get_guidestar(core, guidS)
        metric_t = adaptiveOpt.metric_better_r(guideStar_t, integR)
        logging.debug(f"Metric ={diff}")
        
        #calculates the terms for the next iteration
        g_t = g_0*metric_t/M0
        logging.debug(f"gain ={g_t}")
        a_dash_t  = a_dash_t - g_t*diff*D_t
        
        #saves important data
        metrics.append(metric_t)
        polynomialSeries.append(a_dash_t)
        iterations.append(iteration)    
        logging.debug(f"Iteration {i}")

        iteration +=1    
        
    logging.info("Iterations Finished :3")

    if preview: 
        logging.info("Plotting Results")
        
        guideStar = adaptiveOpt.get_guidestar(core, guidS)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(guideStar, cmap='mako')
        ax2.scatter(iterations, metrics, marker='d', color='#B40424')
        ax2.set_title('Metric progression',fontsize=20)
        plt.show()
    
    #slm.close()
    return phaseMask_t, iterations, metrics
