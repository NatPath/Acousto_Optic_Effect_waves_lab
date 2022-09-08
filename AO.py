import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit

from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist

def linear(x,a,b):
    return a*x+b

def fourier_transform_image(image_location):
    im= imread(image_location)
    im_grey= rgb2gray(im)
    plt.figure(num=None,figsize=(8,6),dpi=80)
    plt.imshow(im_grey,cmap='gray')

    im_grey_fourier = np.fft.fftshift(np.fft.fft2(im_grey))
    plt.figure(num=None,figsize=(8,6),dpi=80)
    plt.imshow(np.log(abs(im_grey_fourier)),camp='gray')

#%% 2- Measurments
#%%1 - 9
h =62e-3 # [m] height of cell
v_sound_theoretical = 1144 # [m/sec] speed of sound in ethyanol
wave_length_laser = 6328e-10 # [m] wave length of lazer needs to be assigned wit hreal value
f_from_n_for_given_wave_length = lambda n : v_sound_theoretical*n/(2*h) #lambda function to calculate f(n) for standing wave in the cell
n_from_f = lambda f : np.ceil(2*f*h/v_sound_theoretical)
f =f_from_n_for_given_wave_length(n_from_f(2e6))  #[Hz] frequency in the 2MHZ range; 217 will resault in 2002000Hz=2.002MHz
x_avg = 1 # [m]

#7
# f_array = [f_from_n_for_given_wave_length(n) for n in range(n_from_f(1e9),n_from_f(1e9)+10)] #array of 10 standing wave frequencies from 1MHz 
f_array = np.array([])*1e9 #Hz
f_err=0 #Hz needs to be assigned with real value
x_avg = np.array([]) #m
x_avg_err=1e-3 #[m] needs to be assigned with real value

fig1= plt.figure("figure 1",dpi=300)
plot1 = plt.errorbar(1/f_array,x_avg,x_avg_err,f_err/(f_array**2),"o",label="data")

# regression
reg1 = linregress(1/f_array,x_avg)
reg_plot = plt.plot(1/f_array,linear(1/f_array,reg1.slope,reg1.intercept),"-.",label="regression")

plt.grid()
plt.ylabel("Average distance between nodes [m]")
plt.xlabel("Time period of sound [seconds]")
plt.legend()
plt.show()

v_sound_found=2*reg1.slope
print("speed of sound found in part A is: " , v_sound_found , "[m/sec]")

#%% 10 - 19
#12 -16
f3=35e-3 # focal point of lense 3
f = 1
x_avg =1

#16 - 19
f_array = np.array([])*1e9 #Hz
f_err=0 #Hz needs to be assigned with real value
x_avg = np.array([]) #m
x_avg_err=1e-3 #[m] needs to be assigned with real value

fig2= plt.figure("figure 2",dpi=300)
plot2 = plt.errorbar(f_array,x_avg,x_avg_err,f_err,"o",label="data")

# regression
reg2 = linregress(f_array,x_avg)
reg_plot = plt.plot(f_array,linear(f_array,reg1.slope,reg1.intercept),"-.",label="regression")

plt.grid()
plt.ylabel("Average distance between nodes [m]")
plt.xlabel("Frequency of sound [Hz]")
plt.legend()
plt.show()

v_sound_found=(f3*wave_length_laser)/reg2.slope
print("speed of sound found in part B is: " , v_sound_found , "[m/sec]")
