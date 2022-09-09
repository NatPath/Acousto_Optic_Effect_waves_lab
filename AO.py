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
    im= imread(image_location)[:,:,:3]
    im= rgb2gray(im)
    plt.figure(num=None,figsize=(8,6),dpi=300)
    plt.imshow(im,cmap='gray')

    im_fourier = np.fft.fftshift(np.fft.fft2(im))
    fig=plt.figure(num=None,figsize=(8,6),dpi=300)
    plt.imshow(100*np.log(abs(im_fourier)),cmap='gray')
    fig.savefig(image_location[:-5]+"_fourier.tif")


def ifourier_transform_image(image_location):
    im= imread(image_location)[:,:,:3]
    im= rgb2gray(im)
    plt.figure(num=None,figsize=(8,6),dpi=300)
    plt.imshow(im,cmap='gray')

    im_fourier = np.fft.fftshift(np.fft.ifft2(im))
    fig=plt.figure(num=None,figsize=(8,6),dpi=300)
    plt.imshow(100*np.log(abs(im_fourier)),cmap='gray')
    fig.savefig(image_location[:-5]+"_fourier.tif")



def linear(x,a,b):
    return a*x+b

def LinearRegPrint(reg):
    print("printing linear regression results:")
    print("(",reg.slope,"-+",2*reg.stderr,")x + (",reg.intercept,"+-",2*reg.intercept_stderr,")")
    print("R^2 is: " , (reg.rvalue)**2)

#%% 2- Measurments
#%%1 - 9
h =62e-3 # [m] height of cell
v_sound_theoretical = 1144 # [m/sec] speed of sound in ethyanol
wave_length_laser = 6328e-10 # [m] wave length of lazer needs to be assigned wit hreal value
f_from_n_for_given_wave_length = lambda n : v_sound_theoretical*n/(2*h) #lambda function to calculate f(n) for standing wave in the cell
n_from_f = lambda f : np.ceil(2*f*h/v_sound_theoretical)

#f =f_from_n_for_given_wave_length(n_from_f(2e6))  #[Hz] frequency in the 2MHZ range; 217 will resault in 2002000Hz=2.002MHz
f=2.3149e6 #[Hz]
f_err=100 #[Hz]
x_avg = 47.33*5.2e-6 # [m]
x_avg_err= 5.2e-6/np.sqrt(13)
v_single= f*x_avg*2
v_err= 2*np.sqrt((x_avg_err*f)**2+(f_err*x_avg)**2)
print("speed of sound from single frequency 2.3149MHz in NF is: ", v_single, "-+",v_err,"[m/sec]")

#7
# f_array = [f_from_n_for_given_wave_length(n) for n in range(n_from_f(1e9),n_from_f(1e9)+10)] #array of 10 standing wave frequencies from 1MHz 

f_array = np.array([ 2.3149,2.3239, 2.3321, 2.3496, #2.3542,
                     2.3633, #2.3725,
                    2.3817, 2.3908])*1e6 #Hz
f_err=0.1 #Hz needs to be assigned with real value
x_avg = np.array([47.33,47, 46+2/3 ,46.5,#47.46,
                  46.428,# 46,
                  46.36, 46.125])*5.2e-6 #m
x_avg_err=5.2e-6/np.sqrt(13) #[m] needs to be assigned with real value

fig1= plt.figure("figure 1",dpi=300)
plot1 = plt.errorbar(1/f_array,x_avg,x_avg_err,f_err/(f_array**2),"o",label="data")

# regression
reg1 = linregress(1/f_array,x_avg)
reg_plot = plt.plot(1/f_array,linear(1/f_array,reg1.slope,reg1.intercept),"-.",label="regression")
plt.grid()
plt.ylabel("x [m]")
plt.xlabel("1/f [seconds]")
plt.legend()
plt.show()

LinearRegPrint(reg1)
v_sound_found=2*reg1.slope
print("speed of sound found in part A is: " , v_sound_found , "-+", 2*reg1.stderr,"[m/sec]")
fig1.savefig("fig/nf_graph")

#%% 10 - 19
#12 -16
f3=300e-3 # focal point of lense 3
f_single = 2.3149e6
f_err=100
pixels=np.array( [ 284, 359, 432, 507, 589, 657])
p_avg_single =np.average(np.diff(pixels)) #[m]
x_avg_single =p_avg_single*5.2e-6 #[m]
x_avg_err=(5.2e-6)/np.sqrt(6)
print("Average distance for 2.3149MHz is: ", x_avg_single,"-+",x_avg_err)
v=f_single*f3*wave_length_laser/x_avg_single
v_err=f3*wave_length_laser*np.sqrt((f_err/x_avg_single)**2+(x_avg_err*f_single/(x_avg_single)**2))
print("speed of sound from single frequency 2.3149MHz in FF is: ", v, "-+",v_err,"[m/sec]")


#16 - 19
f_array = np.array([2.3149,2.0049,1.9049, 1.1349, 1.2349,1.3349, 1.4249, 1.5249, 1.6249, 1.7249, 1.8249, 1.9249 ])*1e6 #Hz
f_err=100 #Hz needs to be assigned with real value
x_avg = np.array([74.6,65,62,37, 40 , 43, 47, 50, 53, 56, 60, 63])*5.2e-6 #m
x_avg_err=5.2e-6 #[m] needs to be assigned with real value

fig2= plt.figure("figure 2",dpi=300)
plot2 = plt.errorbar(f_array,x_avg,x_avg_err,f_err,"o",label="data")

# regression
reg2 = linregress(f_array,x_avg)
LinearRegPrint(reg2)
reg_plot = plt.plot(f_array,linear(f_array,reg2.slope,reg2.intercept),"-.",label="regression")

plt.grid()
plt.ylabel("x[m]")
plt.xlabel("f[Hz]")
plt.legend()
plt.show()

v_sound_found=(f3*wave_length_laser)/reg2.slope
v_err=f3*wave_length_laser*reg2.stderr/(reg2.slope)**2
print("speed of sound found in part B is: " , v_sound_found ,"-+",v_err, "[m/sec]")

fig2.savefig("fig/ff_graph")

#%% fourier analysis of photos

ifourier_transform_image("images/FF_23149.tif")
fourier_transform_image("images/NF_23149.tif")

ifourier_transform_image("images/NF_2314_fourier.tif")