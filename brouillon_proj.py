# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:34:39 2016

@author: perfection
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
from numpy.fft import fft
#from winsound import PlaySound
#import winsound
from scipy.io.wavfile import read as wavread

#%% Mise en place de filtre passe bande

H1 = 200
H2 = 400
H3 = 800
H4 = 1600
H5 = 3200

#def filtre_1:

#%% Mise en place de filtre passe haut
x1,original_signal = wavread('signal43.wav') # for example ; x1 is the size of the array, original_signal is the data itself
x2,beat = wavread('tap43.wav') # for example ; same here

#print(original_signal)
#print(beat)

plt.figure(1)
plt.plot(original_signal)
plt.show()
