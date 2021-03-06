# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:34:39 2016

@author: perfection
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
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

def transfert_passe_bas(f,fc):
    dm = 1 - (f/fc)**2
    return 1/dm

    
#%% Mise en place de filtre passe haut
x1,original_signal = wavread('signal43.wav') # for example ; x1 is the size of the array, original_signal is the data itself
x2,beat = wavread('tap43.wav') # for example ; same here

# Implementation of Scheirer filterbank
# For now, returns nothing, only does tests and print the magnitude frequency response of each band
# The basic frequence still has to be determined
# The functions of scipy.signal work with normalized frequencies, whereas we want to print real frequencies.
# I think we should return the list of (ai,bi) pairs
def filter_bank_1():
    res = []
    fig = plt.figure()
    plt.title('Digital filter frequency response')
    plt.ylabel('Magnitude response (dB)', color='b')
    plt.xlabel('Frequency (Hz)')
    ax1 = fig.add_subplot(111)
    
    samp_v = 44000 # sampling frequence
    v = 200/samp_v # reduced frequence
    # 1st band
    # order 6 ;  max difference in passing band = 3 ; max distance in stop band = 40 ; one scalar or an length-2 array giving critical frequencies ; btype = low pass
    b1,a1 = sig.ellip(6, 3, 40, v, btype='low',analog=True)
    freq1, response1 = sig.freqs(b1,a1)
    ampl1 = np.abs(response1)
    plt.semilogx(freq1/(2*np.pi), 20*np.log10(ampl1), 'r-')
    plt.axis([0.,freq1[-1]/(2*np.pi),-40.,3.])
    
    # 3rd band
    # order 6 ;  max difference in passing band = 3 ; max distance in stop band = 40 ; one scalar or an length-2 array giving critical frequencies ; btype = low pass
    b3,a3 = sig.ellip(6, 3, 40, [2*v,4*v], btype='bandpass',analog=True)
    freq3, response3 = sig.freqs(b3,a3)
    ampl3 = np.abs(response3)
    plt.semilogx(freq3/(2*np.pi), 20*np.log10(ampl3), 'g-')
    plt.axis([0.,freq3[-1]/(2*np.pi),-40.,3.])
    
    # 5th band
    # order 6 ;  max difference in passing band = 3 ; max distance in stop band = 40 ; one scalar or an length-2 array giving critical frequencies ; btype = low pass
    b5,a5 = sig.ellip(6, 3, 40, [8*v,16*v], btype='bandpass',analog=True)
    freq5, response5 = sig.freqs(b5,a5)
    ampl5 = np.abs(response5)
    plt.semilogx(freq5/(2*np.pi), 20*np.log10(ampl5), 'b-')
    plt.axis([0.,freq5[-1]/(2*np.pi),-40.,3.])


    plt.show()
    fig = plt.figure()
    plt.title('Digital filter frequency response')
    plt.ylabel('Magnitude response (dB)', color='b')
    plt.xlabel('Frequency (Hz)')
    ax1 = fig.add_subplot(111)
    
    # 2nd band
    # order 6 ;  max difference in passing band = 3 ; max distance in stop band = 40 ; one scalar or an length-2 array giving critical frequencies ; btype = low pass
    b2,a2 = sig.ellip(6, 3, 40, [v,2*v], btype='bandpass',analog=True)
    freq2, response2 = sig.freqs(b2,a2)
    ampl2 = np.abs(response2)
    plt.semilogx(freq2/(2*np.pi), 20*np.log10(ampl2), 'r-')
    plt.axis([0.,freq2[-1]/(2*np.pi),-40.,3.])

    # 4th band
    # order 6 ;  max difference in passing band = 3 ; max distance in stop band = 40 ; one scalar or an length-2 array giving critical frequencies ; btype = low pass
    b4,a4 = sig.ellip(6, 3, 40, [4*v,8*v], btype='bandpass',analog=True)
    freq4, response4 = sig.freqs(b4,a4)
    ampl4 = np.abs(response4)
    plt.semilogx(freq4/(2*np.pi), 20*np.log10(ampl4), 'g-')
    plt.axis([0.,freq4[-1]/(2*np.pi),-40.,3.])
         
    # 6th band
    # order 6 ;  max difference in passing band = 3 ; max distance in stop band = 40 ; one scalar or an length-2 array giving critical frequencies ; btype = low pass
    b6,a6 = sig.ellip(6, 3, 40, 16*v, btype='high',analog=True)
    freq6, response6 = sig.freqs(b6,a6)
    ampl6 = np.abs(response6)
    plt.semilogx(freq6/(2*np.pi), 20*np.log10(ampl6), 'b-')
    plt.axis([0.,freq6[-1]/(2*np.pi),-40.,3.])
       
    plt.show()

filter_bank_1()

# Plotting the signal and the beat, in order to see what we should have
#plt.figure(1)
#plt.plot(original_signal)
#plt.plot(beat)
#plt.show()
