########
"""
Auxiliary Legacy Functions for SPIM2
By Luis Bastidas, Artemis Castelblanco
Versión: 0.2.20240305
"""
#######

#######
#Dependencies
#######
import cv2
import numpy as np
import scipy.ndimage
from skimage import io
#TODO Cheack slmpy Compatibility
import slmpy
from tqdm import tqdm


######
"""
(Legacy) Zernike Functions
"""
######
class zernike:
    """_summary_Radial
    Calcula el polinomio radial asociado a los números n,m para un disco unitario de radio rho.
    """
    def Radial(n,m,rho, resX = 1920, resY=1152):
        #Iniciación de varibles
        R_nm = np.zeros([resY,resX])
        
        #Se tiene en cuenta solamente los polinomios donde n - m es un número par, por definición..
        dif = int((n-m)/2)
        
        #Se itera sobre todo la imagen.
        for i in tqdm(range(resX)):
            for j in range(resY):
                #Se aplica propiedades para optimizar tiempos de computación. 
                if n == m:
                    R_nm[j,i] = rho[j,i]**m
                elif (n-m)%2 == 0:
                    for k in range(dif+1):
                        #Se desarrolla la sumatoria.
                        R_nm[j,i] +=  (-1)**k * np.math.factorial(n-k)* rho[j,i]**(n-2*k)/(np.math.factorial(k)*np.math.factorial(int((n+m)/2-k))*np.math.factorial(int((n-m)/2-k))) 
        
        #Se retorna el polinomio radial.
        return R_nm


    """_summary_
    Calcula el polinomio de Zernike asociado a los números n,m.
    n,m: números enteros que cumnplen n >= m. m puede tomar valores entre -n y n.
    nx,ny: Desplazamiento del centro del disco unitario en dirección x y respectivamente. Esto se utiliza para ajustar el centro del SLM con el centro del sistema óptico. Por defecto, un desplazamiento nulo.
    resX,resY: Resolución en x y de la pantalla a proyectar el polinomio. Por defecto, se usan las dimensiones del SLM.
    theta: Ángulo de inclinación (en grados) del SLM con respecto al sistema óptico. Implica un escalamiento en x de 1/cos(theta). Por defecto cero.
    """
    def polynomial(n,m,nx = 0,ny = 0,resX = 1920,resY = 1152, theta = 0):
        #Iniciación de variables. Se define el grid de la pnatalla con respecto a resX y rexY. Se escala el eje x con respecto al ángulo theta.
        dx = np.linspace(-1,1,resX)
        dy = np.linspace(-1,1,resY)/np.cos(theta*np.pi/180)
        #Se parametriza las coordenadas rho y phi en el disco unitario. Se aplica el desplazamiento con respecto a nx, ny.
        X,Y = np.meshgrid(dx,dy)
        rho = np.sqrt((X-nx)**2 + (Y-ny)**2)
        phi = np.arctan2(Y-ny,X-nx)
        
        #Se define el polinomio de Zernike impar y par por el signo de m.
        if m < 0:
            m = abs(m)
            R_nm = zernike.Radial(n,m,rho)
            Z_nm = (R_nm*np.sin(m*phi)+1)*255/2
            #Z_nm = R_nm*np.sin(m*phi)
            #Z_nm[rho>1] = 0.0
        else:
            R_nm = zernike.Radial(n,m,rho)
            Z_nm = (R_nm*np.cos(m*phi)+1)*255/2
            #Z_nm = R_nm*np.cos(m*phi)
            #Z_nm[rho>1] = 0.0

        return Z_nm


    """_summary_VectorZ
    Define un arreglo de polinomios de Zernike con respecto al número n. Por defecto n = 2. El arreglo de salida es de tamaño N = (n+1)(n+2)/2.
    """
    def Vector_Z(n=2,nx=0,ny=0):
        #Iniciación de variables.
        N = (n+1)*(n+2)/2 #Tamaño del vector
        Z = []
        
        #Se generan los polinomios de Zernike por nivel de n. Estos están organizados con respecto a los indices OSA/ANSI.
        for i in range(n+1):
            m = np.arange(2*i+1)-i
            for j in m:
                if (i - j)%2 == 0:
                    #print(i,j)
                    Z.append(zernike.polynomial(i,j,nx,ny))
        return np.array(Z)


    """_summary_Mascara_de_fase
    #Realiza la máscara de fase deseada a partir del vector de coeficientes y el vector de polinomios de Zernike.
    """
    def Mascara_de_fase(a,Z):
        #Iniciación de variables.
        result = np.zeros(Z.shape[1:])
        
        #Se hace la suma producto de cada polinomio con su coeficiente.
        for i in range(Z.shape[0]):
            result += a[i]*Z[i]
        
        #Se transforma la imagen a valores enteros entre 0 y 255 para ser proyectadas en el SLM.
        # if(result.min() != result.max()):
        #     result = ((result - result.min())*255/(result.max()-result.min())).astype(int)
        #result[rho>1] = 0
        return result


    """_summary_new_vector
    #Opera el vector de coeficientes para definir el vector de la siguiente iteración.
    # a_t: Vector de coeficientes de la iteración t.
    # M: Métrica de la PSF de la iteración t.
    # g: Término de ganancia del proceso de optimización. Por defecto g = 0.03."""
    def new_vector(a_t, M,g = 0.03):
        #Iniciación de variables. Se define un vector aleatorio por iteración.
        Dt = np.random.rand(a_t.shape[0])*2-1
        return a_t - g*M*Dt


    """_summary_Defocus

        Raises: Intensity, Sign
            ValueError: _description_

        Returns: Defocus with given intensity, positive or negative
            _type_: Z.asytype __
    """
    def Defocus(I = 1, signo = 1,nx=0,ny=0):
        #Iniciación de variables. 
        
        #Se define el defocus con el polinomio dew Zernike n=2, m=0.
        Z = ((zernike.polynomial(2,0,nx,ny)+1)*255/2)
        
        #Se aplica la escala por intensidad.
        Z = I*Z
        
        #Se hace la trampa de proyectar mayor intensidad.
        while Z.max() >254:
            
            Z[Z > 254] = Z[Z > 254] - 255
        
        #Se aplica el cambio de signo para un defocus negativo
        if signo != 1:
            Z = -1*Z +254
        #Z[rho>1] = 0
        return Z.astype(int)


    """_summary_mover
    #Permite mover las máscaras de fase a un pixel específicado por [nx, ny]. El centro de la imágen corresponde a [0,0]. Esta función se utiliza para enconttrar el centro del sistema óptico con repecto al SLM.
    """
    def mover(Z,nx,ny, resX = 1920, resY=1152):
        #Se inicializa la matriz de ceros que se modificará según la posición a la que se mueva la imágen.
        nuevo = np.zeros(Z.shape)

        #Dependiendo de la diección a la que se quiera mover la imágen, se aplica la transformación pertinente para cada cuadrante.
        if nx >= 0 and ny >= 0:
            nuevo[ny:,nx:] = Z[:resY-ny,:resX-nx]
        elif nx < 0 and ny >= 0:
            nuevo[ny:,:resX-abs(nx)] = Z[:resY-ny, abs(nx):]
        elif nx < 0 and ny < 0:
            nuevo[:resY-abs(ny),:resX-abs(nx)] = Z[abs(ny):,abs(nx):]
        else:
            nuevo[:resY-abs(ny),nx:] = Z[abs(ny):,:resX-nx]
        return nuevo

######
"""
SLM Functions
"""
######


class slmpack:
    

    """_connects to the SLM, Must be Called once on startup_

        Raises: NONE

        Returns: NONE
    """
    def startSLM():
        #TODO Cheack slmpy Compatibility
        #slm = slmpy.SLMdisplay(monitor = 1)
        pass
    

    def shift_image(I, dx, dy):
        """_Shifts the center of the I image by dx and dy using the numpy roll function_

        Raises: I image array; dx, dy shift integers 
            ValueError: _description_

        Returns: Shifted array
            _type_: _description_
        """        
        I = np.roll(I, dy, axis=0)
        I = np.roll(I, dx, axis=1)
        if dy>0:
            I[:dy, :] = 0
        elif dy<0:
            I[dy:, :] = 0
        if dx>0:
            I[:, :dx] = 0
        elif dx<0:
            I[:, dx:] = 0
        return I


######
"""
Image acquisition related Function
"""
######
class lumiere:
    def take_Image(core):
        """
        #uses pycromanager to take an image 

        Parameters:
            Core, the microscope

        Returns:
            A 2D numpy array containing the values from the image.
        """
        core.snap_image()
        tagged_image = core.get_tagged_image()
        imageH = tagged_image.tags['Height']
        imageW = tagged_image.tags['Width']
        image = tagged_image.pix.reshape((imageH,imageW))
        return image 
    
    def take_Tiff_image (core): 
        #TODO Make function that takes an image and keeps it as a TIFF
        pass


######
"""
(Legacy) Image Metrics
"""
######

class metrics:
    """_summary_metric_r_power_integral
    #Calcula la métrica de una PSF de 2 dimensiones. Código obtenido de Vladimirov et al. (2019).
    """
    def metric_r_power_integral(img, integration_radius=20, power=2):
        """Metric of PSF quality based on integration of image(r) x r^2 over a circle of defined radius. 
        From Vorontsov, Shmalgausen, 1985 book. For best accuracy, img dimensions should be odd, with peak at the center.
        Parameters:
            img, a 2D image with PSF peak at the center
            integration_radius, for the circle of integration, default 20.
            background_subtract, value of camera offset (0 by default)
            """
        img = (img-img.min())/(img.max()-img.min())
        h, w = img.shape[0], img.shape[1]
        if np.min(img.shape) < 2 * integration_radius:
            raise ValueError("Radius too large for image size")
        else:
            # center = [int(w / 2), int(h / 2)]
            # center of mass center, does not tolerate > 1 beads in FOV!
            bg = np.percentile(img, 99)
            roi_binary = np.zeros(img.shape)
            roi_binary[img > bg] = 1
            cmass = scipy.ndimage.measurements.center_of_mass(roi_binary)
            x_center, y_center = cmass[1], cmass[0]
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x + 0.5 - x_center) ** 2 + (y + 0.5 - y_center) ** 2)
            mask = (dist_from_center <= integration_radius).astype(int)
            metric = np.sum(img * mask * (dist_from_center ** power)) / np.sum(mask)
        return metric

    """_summary_TIFMetrica1
    #Función para usar las diferentes imagenes de tif"""
    def TIFMetrica1(tifname):
        # read input image as grayscale
        imf = io.imread(tifname)
        # convert the grayscale to float32
        imf = np.float32(imf) # float conversion
        n=(imf.shape[0])
        entropys=np.zeros(n)
        #Recorrido de cada imagen para la prueba
        for i in range(n):
            dct=cv2.dct(imf[i], cv2.DCT_INVERSE)
            entropys[i]= metrics.DCTS(dct)     
        return entropys

    """_summary_DCTS
    #Métrica de la entropía de Shannon con DCT
    #Parametro tifname. Nombre del archivo de la imagen en formato .tiffcula la métrica de una PSF de 2 dimensiones. Código obtenido de Vladimirov et al. (2019).
    """     
    def DCTS(tifname):
        # read input image as grayscale
        imf = io.imread(tifname)
        # convert the grayscale to float32
        imf = np.float32(imf) # float conversion
        #Transformada Discrete Cosine Transform (DCT).
        I=cv2.dct(imf, cv2.DCT_INVERSE)
        #Height, Weigth
        H, W = I.shape
        #support PSFSupportdiameter: 2.8 pixeles en el diametro de Airy, 3 aproximado
        #r0: support radius. Cada coordenada transformada tiene su propio r0.
        #r0x=ancho/1.5pixeles, r0y=alto/1.5pixeles.
        r0x=int(W/3)
        r0y=int(H/3)
        # NORM 2 of dct
        l2=cv2.norm(I)
        suma=0
        for i in range(r0x):
            #Se recorren solo los pixeles de la transformada que están en la línea
            #Se descartan los numeros de onda altos en x y en y ya que son ruido.
            r0yf=r0y-i*(r0y/r0x)
            for j in range(int(r0yf)):
                if I[i][j]==0:
                    suma+=0
                else:
                    suma+=abs(I[i][j]/l2)*np.log2(abs(I[i][j]/l2))
        return -(1/(r0x*r0y))*suma