# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:05:14 2023

@author: MayaYanko
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Sinewaves horizontal – for tasks 1&2
N = 256
x = np.linspace(-np.pi,np.pi, N)
sine1D = 128.0 + (127.0 * np.sin(x * 8.0))
sine1D = np.uint8(sine1D)
sine2D = np.tile(sine1D, (N,1))
print(sine2D.shape)
# plt.imshow(sine2D, cmap='gray')
# plt.show()

basic_path = 'images/'
# Barbara image (use only green channel) – for tasks 1, 2, 3, 4
Barbara = cv2.imread(basic_path+'Barbara.jpg')[:,:,::-1]
# plt.imshow(Barbara)
# plt.show()
green_Barbara = Barbara[:,:,1]
# plt.imshow(green_Barbara, cmap='gray')
# plt.show()

#low-contrast-image – for tasks 5
low_contrast_lena = cv2.imread(basic_path+'low-contrast-image.jpg')[:,:,::-1]
# plt.imshow(low_contrast_lena, cmap='gray')
# plt.show()


def plot_image(image, title):
    print(image.shape)
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.show()
    
def sample_image(image, rate):
    ezer_image = image.copy()
    ezer_image = ezer_image[::rate,::rate]
    return ezer_image

def quantize_image(image, start_rate, end_rate):
    # ezer_image = image.copy()
    # final_levels = int(start_rate/end_rate)
    # for i in range(final_levels):
    #     ezer_image[(ezer_image>i*final_levels) & (ezer_image<=(i+1)*final_levels)] = i  
    return np.floor_divide(image, 16)

def gamma_correction(image, gamma, gray_levels): 
    return ((image/(N-1))**gamma)*(gray_levels-1)

def image_histogram(image, gray_levels): 
    histogram, bin_edges = np.histogram(image, bins=gray_levels)
    plt.plot(bin_edges[0:-1], histogram)
    plt.xlim([0,gray_levels])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def contrast_stretching(image, gray_levels):
    return (image-image.min())/(image.max()-image.min())*(gray_levels-1)
    
if __name__ == '__main__':
    # Task 1: write code that takes an image and sample it in a rate of 2^n (every second pixel, every forth pixel till you can’t)
    # for rate in map(lambda x: 2**x, range(1,8)):
    #     ezer_image = sample_image(sine2D, rate)  
    #     plot_image(ezer_image, f'sample image in a rate={rate}')
    # for rate in map(lambda x: 2**x, range(1,8)):
    #     ezer_image = sample_image(green_Barbara, rate)
    #     plot_image(ezer_image, f'sample image in a rate={rate}')

    # Task 2: write code that take 8 bits image and quantize it to 4, 2 and 1 bit.
    # for rate in map(lambda x: 2**x, [4,2,1]):
    #     ezer_image = quantize_image(sine2D, N, rate)  
    #     plot_image(ezer_image, f'quantize image in a rate={rate}')
    # for rate in map(lambda x: 2**x, [4,2,1]):
    #     ezer_image = quantize_image(green_Barbara, N, rate)  
    #     plot_image(ezer_image, f'quantize image in a rate={rate}')
    
    # Task 3: write code that does gamma correction for an image
    # for gamma_to_correct in np.arange (0.2, 2.5, 0.2):
    #     ezer_image = gamma_correction(green_Barbara, gamma_to_correct, N)
    #     plot_image(ezer_image, f'gamma correction\ngamma={gamma_to_correct}')
    
    # Task 4: write code that calculates the histogram of an image
    image_histogram(green_Barbara, N)  
    plt.close()
    
    # Task 5: write code that does contrast stretching for an image
    image_histogram(low_contrast_lena, N)  
    plt.close()
    high_contrast_lena = contrast_stretching(low_contrast_lena, N)  
    image_histogram(high_contrast_lena, N)  