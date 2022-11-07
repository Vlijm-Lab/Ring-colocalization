import os
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from ring_analysis.functions import swap_channels

def display_stack(stack, pixel_size, color1, color2, channel1, channel2):
    '''
    Displays a dual layer image in three images: combined using the given colormaps, first layer in grayscale and second layer in grayscale

            Parameters:
                    stack (np.ndarray): 3D array consisting of a dual layer image.
                    pixel_size (float): pixel size in nm.
                    color1 (str): colormap for first layer of the image stack.
                    color2 (str): colormap for second layer of the image stack.
                    channel1 (str): name of first layer of image stack.
                    channel2 (str): name of second layer of image stack.

            Returns:
                    None
    '''
    
    fig, ax = plt.subplots(1,3, figsize = (15,5))
    
    ax[0].imshow(colormap(stack[0], color1)+colormap(stack[1], color2))
    ax[0].set_title('Combined')
    ax[0].add_artist(ScaleBar(pixel_size, 
                              'nm', 
                              location = 'lower right',
                              color = 'w',
                              box_alpha = .0,
                              length_fraction=0.25)); 
    
    ax[1].imshow(stack[0], cmap = 'gray')
    ax[1].set_title(channel1)
    ax[1].add_artist(ScaleBar(pixel_size, 
                              'nm', 
                              location = 'lower right', 
                              color = 'w', 
                              box_alpha = .0, 
                              length_fraction=0.25)); 
    
    ax[2].imshow(stack[1], cmap = 'gray')
    ax[2].set_title(channel2)
    ax[2].add_artist(ScaleBar(pixel_size, 
                              'nm', 
                              location = 'lower right', 
                              color = 'w', 
                              box_alpha = .0, 
                              length_fraction=0.25)); 
 
    plt.show()
    
def colormap(image, color):
    '''
    Changes the colormap of the grayscale image (2D) to one of the six basic colors.

            Parameters:
                    image (np.ndarray): 2D array containing grayscale image values.
                    color (str): One of the six basic colors, options: 'red', 'blue', 'green', 'yellow', 'magenta' and 'cyan'.

            Returns:
                    result (nd.ndarray): 3D array, containing the 2D data of `image` mapped to the given colormap. 
                                         Resulting data will be max normalized ([0,1] for each color channel).
    '''
        
    h,w = image.shape
    result = np.zeros((h,w,3))
    
    norm_image = image/np.max(image)
    
    if color == 'red' or color == 'magenta' or color == 'yellow':
        result[:,:,0] = norm_image
    if color == 'green'or color == 'cyan' or color == 'yellow':
        result[:,:,1] = norm_image
    if color == 'blue' or color == 'magenta' or color == 'cyan':
        result[:,:,2] = norm_image
    
    return result

def display_all_images(parameters):
    '''
    Display the dual stack tiff images in the raw image folder.

            Parameters:
                    parameters (Parameters): Class containing all parameters or the peak to peak analysis.

            Returns:
                    None
    '''
    
    image_list = os.listdir(parameters.img_dir)
    
    for image_path in image_list:
        tif_stack = tif.imread(f'{parameters.img_dir}/{image_path}')
        
        if not parameters.top_layer_dominant:
            tif_stack = swap_channels(tif_stack)
            
        display_stack(tif_stack, pixel_size = parameters.pixel_size, color1 = parameters.color1, color2 = parameters.color2, channel1 = parameters.channel1, channel2 = parameters.channel2)