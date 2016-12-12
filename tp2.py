# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
from numpy.fft import fft
from numpy.fft import ifft
import scipy.signal as sig

plt.close('all');

#%% filtre du premier ordre: reponse impulsionnelle
N = 128;
x = np.zeros((N,), dtype='f');
x[0]=1;
B = [1];
a = 0.8; #essayer d'autres valeurs
A = [1, -1*a];
y = sig.lfilter(B,A,x);
plt.stem(y); 

#%% filtre du premier ordre: visualisations
zpk = sig.tf2zpk(B,A);
z = zpk[0];
p = zpk[1];
plt.figure(2);
plt.scatter(np.real(p),np.imag(p))
plt.scatter(np.real(z),np.imag(z))

w, h = sig.freqz(B,A)
fig = plt.figure(3);
plt.title('Digital filter frequency response');
ax1 = fig.add_subplot(111);
plt.plot(w, 20 * np.log10(abs(h)), 'b');
plt.ylabel('Amplitude [dB]', color='b');
plt.xlabel('Frequency [rad/sample]');
ax2 = ax1.twinx();
angles = np.unwrap(np.angle(h));
plt.plot(w, angles, 'g');
plt.ylabel('Angle (radians)', color='g');
plt.grid();
plt.axis('tight')
plt.show();

#%% filtre du premier ordre : FFT, vérifier que les valeurs en f=0 et f=1/2
# sont : 1/(1-a) et 1/(1+a) 
h = np.power(a,range(0,N));
plt.figure(1); plt.plot(h);
H = fft(h);
plt.figure(4); plt.plot(abs(H));
plt.show();

#%% filtre du premier ordre : filtrage d'une sinusoide
Fe = 32;
N = 128; 
t = np.arange(N)/Fe;
f0 = 3;
x = np.sin(2*np.pi*f0*t);
y1 = sig.lfilter(B,A,x);
y2 = sig.lfilter(h,[1],x);
plt.figure(5);
plt.plot(y1); plt.plot(y2); plt.figure(6); plt.plot(y1-y2);
plt.figure(7);
plt.plot(abs(fft(x))); plt.plot(abs(fft(h))); plt.plot(abs(fft(y1)));
plt.show();

#%% convolution circulaire
y3=np.real(ifft(fft(x)*fft(h)));
plt.figure(8); plt.plot(y3); plt.plot(y1); 
plt.show();

#%% gain et dephasage entree sortie
plt.figure(9); plt.plot(x); plt.plot(y1);
plt.show();

#%% train d'impulsions rectangulaires
x = np.sign(np.sin(2*np.pi*f0*t));
y = sig.lfilter(B,A,x);
plt.figure(10);
plt.plot(x); plt.plot(y);
plt.show();

f1=0.5;
x=np.sign(np.sin(2*np.pi*f1*t));
y=(1-a)*sig.lfilter(h,[1],x);
plt.figure(11);
plt.plot(t,y,label='y');
plt.plot(t,x,label='x');
plt.ylim([-1.2, 1.2]);
plt.show();
