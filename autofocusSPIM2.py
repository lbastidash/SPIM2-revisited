#Autofoco SPIM 2
#Autorxs: Cristhian Perdomo, Luis Bastidas, Alex Artemis Castelblanco
#cd.perdomo10@uniandes.edu.co
#l.bastidash@uniandes.edu.co
#Versión: 2024/02/29

#Aqui importar las funciones necesitadas de FuncionesSPIM

from FuncionesSPIM import DCTS

import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import pytic
from time import sleep

from random import randint
from pycromanager import Bridge
from pycromanager import Acquisition, multi_d_acquisition_events
import numpy as np
import time
#import napari

import sys

from datetime import datetime
import os

import cv2 as cv

from PIL import Image, ImageSequence

'''
You may notice that the function names are have changed slightly from the example above to the onces listed here. 
Specifically, "snapImage" was called as "snap_image". This is because Pycro-Manager automatically converts from the 
Java convention of "functionsNamesLikeThis()" to the Python convention of "functions_named_like_this()". It is possible to 
change this behavior when creating the bridge with Bridge(convert_camel_case=False)
'''

bridge = Bridge()
core = bridge.get_core() #No borrar!!

#'''
properties = core.get_device_property_names("HamamatsuHam_DCAM")
for i in range (properties.size() ): 
    prop = properties.get(i)
    val = core.get_property("HamamatsuHam_DCAM", prop)
    print ("Name: " + prop + ", value: " + val)
#'''

'''
COM5
COM1
HamamatsuHam_DCAM
Sutter MPC
Sutter MPC Z stage
Sutter MPC XY stage
Oxxius LaserBoxx LBX or LMX or LCX
GenericSLM
Core
'''

core.set_property("Sutter MPC", 'Step Size', '1')

def InitPos (): #lee la posición del micromanipulador y la devuelve

    x_pos = core.get_property('Sutter MPC', 'Current X')
    y_pos = core.get_property('Sutter MPC', 'Current Y')
    z_pos = core.get_property('Sutter MPC', 'Current Z')
    return float(x_pos), float(y_pos),float(z_pos)

#TODO poner comentario no  cutre 
def GoToX (Posicion: float): #Mueve la posicion de la microcrontrolacion 
    core.set_property('Sutter MPC', 'Current X', str(Posicion))


def Autofocus1(fileName: str, folderPath: str):
    # - Inicializacion del controlador -----------------------------------
    tic = pytic.PyTic()

    # Connect to first available Tic Device serial number over USB
    serial_nums = tic.list_connected_device_serial_numbers()
    tic.connect_to_serial_number(serial_nums[0])

    # Load configuration file and apply settings
    tic.settings.load_config('config.yml')
    tic.settings.apply()                             

        # - Motion Command Sequence ----------------------------------
        # Zero current motor position
    tic.halt_and_set_position(0)
    # Energize Motor
    tic.energize()
    tic.exit_safe_start()
    #Inicializar el objetivo de iluminación en una zona. Posicion inicial
    PI=0
    #Posición inicial de la muestra
    xI, yI, zI =   InitPos() 

    #Distancia estandar con la que se moverá el objetivo de iluminación. 1+/-0.02micras
    d=20
    stepSize =  1 #Cambio para el micromanipulador, 1micra
    #Posicion final
    PF=400
    xF=PF/d 
    #Array que tendra las posiciones predeterminadas para las metricas
  
    x1= np.arange(xI-xF, xI, stepSize)
    x2=  np.arange(xI, xI+xF+stepSize, stepSize)
    x= np.concatenate((x1,x2))
    p1= np.arange(-400,PI,d)
    p2= np.arange(PI,400+d,d)
    p= np.concatenate((p1, p2))

    #Array que tendra los valores de metricas para compararlas 
    m=np.zeros(p.size)
    #Empieza desde -500, para moverse a -400 asegurando linealidad.
    tic.set_target_position(-500)
    while tic.variables.current_position != tic.variables.target_position:
        sleep(0.1)

    #Recorrido de las posiciones
    for i in range(p.size):
        #Mueve el micromanipulador y el objetivo de iluminacion
        GoToX(x[i])
        tic.set_target_position(p[i])
        while tic.variables.current_position != tic.variables.target_position:
            sleep(0.1)
        
        #Se hace la toma de datos y se recibe el nombre del archivo de la imagen
        folderPath.replace("\\" , '/') 

        core.snap_image()
        tagged_image = core.get_tagged_image()
        #If using micro-manager multi-camera adapter, use core.getTaggedImage(i), where i is the camera index
        #pixels by default come out as a 1D array. We can reshape them into an image
        pixels = Image.fromarray( np.reshape(tagged_image.pix,newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']]) , mode='I;16'  )
        pixels.save((folderPath + '/'+fileName+str(i)+'.tiff'), "TIFF")  
            
        m[i]=DCTS((os.path.join((folderPath + '/'+fileName+str(i)+'.tiff')) ))
        #Para darle tiempo de tomar la foto
        print(i)
        sleep(0.3)
        #Una vez acaba el recorrido, volver a la posicion inicial
        if i==p.size-1:
            tic.set_target_position(0)
            while tic.variables.current_position != tic.variables.target_position:
                sleep(0.1)

    # De-energize motor and get error status
    tic.enter_safe_start()
    tic.deenergize()
    print(tic.variables.error_status)
        
    plt.scatter(p,m)
    plt.xlabel('Desplazamiento del motor')
    plt.ylabel('Resultado Metrica')
    plt.title('Prueba con Metrica 1')
        
    # Guardar la gráfica en un archivo
    plt.savefig('resultados.png')

filename= input("Coloca el nombre del archivo: ")
folderPath= input("Coloca la ubicación para el archivo: ")
Autofocus1(filename, folderPath)