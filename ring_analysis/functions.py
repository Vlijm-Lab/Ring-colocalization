import numpy as np
import os
import time
import skimage
import tifffile as tif
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.interpolate import interp1d
from scipy.ndimage import binary_fill_holes as bfh
from skimage.morphology import erosion as e
from skimage.morphology import dilation as d
from skimage.measure import label, regionprops

# Length of line profiles calculated and displayed is 21 pixels. Change if structure width is larger.

def display_progress(it, it_max, start_t):
    '''
    Display a progress bar and the estimated time of completion. Function is updated within a loop.

            Parameters:
                    it (int): current itteration of the loop
                    it_max (int): total number of itterations of current loop
                    start_t (int): start time of the system in seconds.

            Returns:
                    None
    '''      
        
    time_taken = int(time.time()- start_t)
    
    if time_taken == 0:
        min_taken = 0
        sec_taken = 0
        
    else:
        min_taken = int(time_taken / 60)
        sec_taken = int(time_taken % 60)
    
    time_left = int((time_taken / (it + 1) * (it_max - (it + 1))))
    
    if time_left == 0:
        min_left = 0
        sec_left = 0
    
    else:
        min_left = int(time_left/60)
        sec_left = int(time_left%60)
    
    print(f'\033[1m[{it+1}/{it_max}] ({int(100*(it+1)/it_max)}%).\033[0;0m Time taken ({min_taken:02d}:{sec_taken:02d}). Time left = ({min_left:02d}:{sec_left:02d})', end='\r')

def ee(image, it = 1):
    '''
    Erode image a given amount of times.

            Parameters:
                    image (np.ndarray): 2D image of single channel pixels (grayscale)
                    it (int): number of times the image needs to be eroded

            Returns:
                    image (np.ndarray): 2D eroded image of single channel pixels (grayscale)img
    '''      
    
    for i in range(it):
        image = e(image)
        
    return image

def ed(image, it = 1):
    '''
    Erode and dilate image a given amount of times.

            Parameters:
                    image (np.ndarray): 2D image of single channel pixels (grayscale)
                    it (int): number of times the image needs to be eroded and dilated

            Returns:
                    image (np.ndarray): 2D eroded and dilated image of single channel pixels (grayscale)img
    '''      
    
    for i in range(it):
        image = e(image)
        image = d(image)
        
    return image

def swap_channels(stack: np.ndarray):
    '''
    Swaps the layers for a 3d np.ndarray of shape (2, ..., ...).

            Parameters:
                    stack (np.ndarray): 3d image stack consisting of 2 layers

            Returns:
                    swapped_stack (np.ndarray):  input stack with reversed order of stacks
    '''      
    
    first_channel, second_channel = stack
    swapped_stack = np.stack([second_channel, first_channel])
    
    return swapped_stack

def line_profile_at_center(image, rot_degrees, x_center, y_center, line_width):
    '''
    Obtain line profile from at predetermined rotation, center and linewidth

            Parameters:
                    image (np.ndarray): 2d image of single pixel values (grayscale)
                    rot_degrees (float): rotation in degrees, determines at what angle the lineprofile will be obtained
                    x_center (float): x-coordinate of the center of the object in image
                    y_center (float): y-coordinate of the center of the object in image
                    line_width (int): linewidth of the perceived line-profile

            Returns:
                    line_profile (np.ndarray):  resulting line profile at the given rotation and of predetermined width
    '''      
    
    rotated_image = scipy.ndimage.rotate(np.copy(image), rot_degrees, reshape = True)
    line_profile = rotated_image[int(y_center) - line_width : int(y_center) + line_width + 1,:]
    
    return line_profile

def outer_peaks(line_profile):
    '''
    Obtain the outer peaks of line profile. Will return zeros if there exist not 2 or more peaks in line profile.

            Parameters:
                    line_profile (np.ndarray): 1d line profile
                    
            Returns:
                    first_peak (int):  index of array line_profile at which the first peak is located (equals 0 if less than 2 peaks)
                    last_peak (int):  index of array line_profile at which the last peak is located (equals 0 if less than 2 peaks)
    '''    
    
    peaks = scipy.signal.find_peaks(line_profile)
    
    if len(peaks[0]) < 2:
        first_peak = 0
        last_peak = 0
    
    else:
        first_peak = peaks[0][0]
        last_peak = peaks[0][-1]
       
    return first_peak, last_peak

def mask_by_largest_obj(image, label_image):
    '''
    Returns image masked by largest object in label_image

            Parameters:
                    image (np.ndarray): 2d image (grayscale)
                    label_image (np.ndarray): 2d label image, consisting of integers
                    
            Returns:
                    masked_image (np.ndarray):  input image masked by largest object of label_image
    '''    
        
    
    bins = np.bincount(label_image.flatten())
    masked_image = image * (label_image == np.argmax(bins[1:]) + 1)
 
    return masked_image

def largest_object(objects):
    '''
    Largest_object calculates the largest object and returns the center coordinates of this object

            Parameters:
                    objects (list of Regionproperties): segmented objects
                    
            Returns:
                    x (float): x-coordinate of the center of largest object of the segmented image
                    y (float): y-coordinate of the center of largest object of the segmented image
    '''    
    
    largest_area = 0
    for obj in objects:
        if obj.area > largest_area:
            largest_object = obj
            largest_area = obj.area

    x, y = largest_object.centroid

    
    return x, y

def segment_objects(image):
    '''
    thresholding, filling holes, dilute and erode to get a binary map of image
    Then withdraw label image, and use reigionprops to get objects

            Parameters:
                    image (np.ndarray): 2d image of single pixel values (grayscale)                    
                    
            Returns:
                    objects (list of Regionproperties): segmented objects
                    label_image (np.ndarray): 2d label image, consisting of integers
    '''   
    
    binary_img = bfh(ed((image>5),1))
    binary_img = d(d(binary_img))
    
    label_img = label(binary_img)
    
    objects = regionprops(label_img) 
    
    return objects, label_img

def horizontal_diameter(image, rot_degrees, line_width, kernel_size = 5, pixel_size = 17, gaus_sigma = 3):
    '''
    Get line profiles from outer peaks of object (ring) of the dominant layer.
    The peak coordinates and rotation angle are returned to use for the non-dominant layer

            Parameters:
                    image (np.ndarray): 2d image of single pixel values (grayscale)
                    rot_degrees (float): rotation in degrees, determines at what angle the lineprofile will be obtained
                    line_width (int): linewidth of the perceived line-profile
                    kernel_size (int): kernel size used for smoothing of 1d array
                    pixel_size (float): pixel size in nm
                    gaus_sigma (float): sigma used for Gaussian Blur
                    
            Returns:
                    x_center (float): x-coordinate of the center of the object in image
                    y_center (float): y-coordinate of the center of the object in image
                    line_profile_copy (np.ndarray): copy of the 2d line profile (width determined by line_width)
                    first_peak (int):  index of array line_profile at which the first peak is located (equals 0 if less than 2 peaks)
                    last_peak (int):  index of array line_profile at which the last peak is located (equals 0 if less than 2 peaks)         
    '''       

    #Rotate image, and make a copy for deducting final line profile
    rotated_image = scipy.ndimage.rotate(np.copy(image), rot_degrees, reshape = True)
    rotated_copy = np.copy(rotated_image)
    
    #Calculate largest segemented object, and label image
    obj, label_image = segment_objects(rotated_image)
    y_center, x_center = largest_object(obj)

    #Mask by largest object / remove secondary objects
    rotated_image = mask_by_largest_obj(rotated_image, label_image)
    
    #Smooth image, otherwise open spots will not be used     
    rotated_image = skimage.filters.gaussian(rotated_image, sigma = gaus_sigma, preserve_range = True)
   
    #Deduct peaks from line profile of width 2*lw+1
    line_profile = rotated_image[int(y_center) - line_width : int(y_center) + line_width + 1,:]
    summed_line_profile = np.sum(line_profile, axis = 0)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_line_profile = np.round(np.convolve(summed_line_profile, kernel, mode='same'))
    
    #Deduct line profile and outer peaks
    line_profile_copy = rotated_copy[int(y_center) - line_width : int(y_center) + line_width + 1,:]
    first_peak, last_peak = outer_peaks(smoothed_line_profile)
    
    return x_center, y_center, line_profile_copy , first_peak, last_peak

def add_lines(line_profile1, line_profile2, left_peak, right_peak, width = 10):
    '''
    Determine line profiles at given peaks: left peak and right peak
    Left line profile gets reversed, then both line profiles get summed

            Parameters:
                    line_profile1 (np.ndarray): line_profile (2d, due to width) of dominant layer
                    line_profile2 (np.ndarray): line_profile (2d, due to width) of non_dominant layer
                    left_peak (int):  index of array line_profile at which the left peak is located 
                    right_peak (int):  index of array line_profile at which the right peak is located 
                    width (int): width of the line profile part which is centered at the dominant peak
                    
            Returns:
                    dominant_line_profile_sum (np.ndarray): sum of left and right line profile of dominant layer
                     non_dominant_line_profile_sum (np.ndarray): lsum of left and right line profile of non dominant layer
                        
    ''' 
    
    #Initialize line profiles
    lp_summed1 = np.sum(line_profile1, axis = 0)
    lp_summed2 = np.sum(line_profile2, axis = 0) 

    zero_array = np.zeros(width*2+1)
    
    #If left peak is lower than width, pad with zeros
    if left_peak <= width:
        
        left_profile1 = np.copy(zero_array)
        left_profile1[width-left_peak:] = lp_summed1[0:left_peak+width+1]
        left_profile1 = left_profile1[::-1]
        
        left_profile2 = np.copy(zero_array)
        left_profile2[width-left_peak:] = lp_summed2[0:left_peak+width+1]
        left_profile2 = left_profile2[::-1]        
    
    else:
        
        left_profile1 = lp_summed1[left_peak-width:left_peak+width+1]
        left_profile1 = left_profile1[::-1]

        left_profile2 = lp_summed2[left_peak-width:left_peak+width+1]
        left_profile2 = left_profile2[::-1]

    
    #If right peak is higher than width, pad with zeros        
    if right_peak >= len(lp_summed1) - width :
        
        right_profile1 = np.copy(zero_array)
        right_profile1[:- (right_peak + 1 -(len(lp_summed1)-width))] = lp_summed1[right_peak-width:]
        
        right_profile2 = np.copy(zero_array)
        right_profile2[:- (right_peak + 1 -(len(lp_summed2)-width))] = lp_summed2[right_peak-width:]  
        
    else:
        
        right_profile1 = lp_summed1[right_peak-width:right_peak+width+1]
        right_profile2 = lp_summed2[right_peak-width:right_peak+width+1]
    
    dominant_line_profile_sum = left_profile1+right_profile1
    non_dominant_line_profile_sum = left_profile2+right_profile2
    
    return dominant_line_profile_sum, non_dominant_line_profile_sum

def rotated_line_profiles(image, line_width, it = 180, step = 1, pixel_size = 17, display = False):   
    '''
    Get line profiles from outer peaks of object (ring) of the dominant layer.
    The peak coordinates and rotation angle are returned to use for the non-dominant layer

            Parameters:
                    image (np.ndarray): 2d image of single pixel values (grayscale)
                    line_width (int): linewidth of the perceived line-profile
                    it (int): number of itterations to loop through
                    step (int): step size, i.e. number of degrees the image is rotated every itteration
                    pixel_size (float): pixel size in nm
                    display (bool): set to `True` to observe all rotations and placed diameters
                    
            Returns:
                    line1 (np.ndarray): sum of line profiles of dominant layer, 
                    centered at the peak of this layer
                    line2 (np.ndarray): sum of line profiles of non-dominant layer, 
                    centered at the peak of the dominant layer
    '''   
    
    #Initialize line profiles of length 20
    line1 = np.zeros(21)
    line2 = np.zeros(21)
    
    #Initialize the figures
    if display:
        rows = it // 3 
        if rows > 1 :
            cols = 3
        else:
            cols = it % 3 + 1
        fig, ax = plt.subplots(rows, cols, figsize = (cols*5, rows * 5))
                             
   
    #loop through all rotations
    for i in range(it):
        
        #Calculate peaks of dominant protein
        rot_angle = i*step
        xo, yo, line_prof1, lpp1, lpp2 = horizontal_diameter(image[0], rot_angle, line_width )
        
        #If there were at least two peaks in the dominant image, get line profile at same place for non-dominant image
        if lpp1+lpp2 != 0:
            line_prof2 = line_profile_at_center(image[1], rot_angle, xo, yo, line_width)
            
            #Add line profiles at peak positions of dominant
            l1, l2 = add_lines(line_prof1, line_prof2, lpp1, lpp2)
            line1 += l1
            line2 += l2

            #Display the dominant image with corresponding diameter
            if display:
                rotated_image = scipy.ndimage.rotate(np.copy(image[0]), i*step, reshape = True)
                ax[i//3, i%3].imshow(rotated_image, cmap = 'hot', interpolation = 'Nearest')
                ax[i//3, i%3].set_title(f'Rotation: {i*step}')
                ax[i//3, i%3].plot((lpp1, lpp2), (int(yo), int(yo)), '-w', linewidth=1)
                legend = ax[i//3, i%3].legend([f'Diam: { (lpp2-lpp1)*pixel_size} nm'], loc="lower right", prop={'size': 10})
                legend.get_frame().set_alpha(None)
                plt.setp(legend.get_texts(), color='b')
            
    if display:
        plt.show()
    
    return line1, line2

def compute_line_profiles(parameters):
    '''
    Computing all the line profile of all images in the directory for all rotations.

            Parameters:
                    parameters (Parameters): Class containing all parameters or the peak to peak analysis.
                    
            Returns:
                    line1 (np.ndarray): sum of line profiles of all stacks of the dominant layer, 
                                        centered at the peak of this layer
                    line2 (np.ndarray): sum of line profiles of all stacks of the non-dominant layer, 
                                        centered at the peak of the dominant layer
    ''' 
    
    #Initialize the line profiles
    line1 = np.zeros(21)
    line2 = np.zeros(21)
    
    #Store starting time
    start_time = time.time()
    
    #Loop over all images
    image_list = os.listdir(parameters.img_dir)
    for image_path, i in zip(image_list, range(len(image_list))):
        
        #load image, and swap channels if needed
        tif_stack = tif.imread(f'{parameters.img_dir}/{image_path}')
        if not parameters.top_layer_dominant:
            tif_stack = swap_channels(tif_stack)
        
        #obtain line profiles for all rotations
        line_profile_1, line_profile_2 = rotated_line_profiles(tif_stack, 2, it = parameters.number_rotations, step = parameters.rotation_step, pixel_size = parameters.pixel_size, display = parameters.display_diameter_calculation)
        line1 += line_profile_1
        line2 += line_profile_2
        
        #Displaying progress bar
        display_progress(i, len(image_list), start_time)
        
    return line1, line2


def peak2peak(parameters, line_profiles):
    '''
    Plot the histograms of the summed line profiles and the interpolated line profile.
    The interpolation used is cubic interpolation.
    The peak to peak distance is computed from the peaks of the interpolated line profiles.

            Parameters:
                    parameters (Parameters): Class containing all parameters or the peak to peak analysis.
                    line_profiles (list): list of two np.ndarrays, containing the summed dominant and summed non-dominant line profiles
                    
            Returns:
                    None
    '''
    
    line1, line2 = line_profiles
    acc = 100
    ip_pres = parameters.pixel_size * acc
          
    #Calculate the interpolated line profile, using cubic interpolation
    f1n = interp1d(np.array(range(len(line1))), line1, kind='cubic')
    f2n = interp1d(np.array(range(len(line2))), line2, kind='cubic')
   
    #Setting the x ranges for the raw line profiles and the interpolated line profiles
    x_orig = np.linspace(0, len(line1)-1, len(line1))
    xnew = np.linspace(0, len(line1) - 1, num = ((len(line1) - 1) * ip_pres + 1))
    
    #Initialize figure, ax2 is used for plotting in one graph
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    #Bar plots of original data
    ax1.bar(x_orig, line1, color = parameters.color1, alpha = 0.1)
    ax1.set_ylim(bottom = 0)
    ax2.bar(x_orig, line2, color = parameters.color2, alpha = 0.1)
    ax2.set_ylim(bottom = 0)
    
    #Interpolated lines
    interpolated_line1 = ax1.plot(xnew, f1n(xnew), parameters.color1, label = parameters.channel1)
    interpolated_line2 = ax2.plot(xnew, f2n(xnew), parameters.color2, label = parameters.channel2)
    
    #Peak locations
    ax1.axvline(np.argmax(f1n(xnew)) / ip_pres, color=parameters.color1, linestyle='--')
    ax2.axvline(np.argmax(f2n(xnew)) / ip_pres, color=parameters.color2, linestyle='--')
    
    #labels & legend
    ax1.set_xlabel('nm')
    ax1.legend(interpolated_line1 + interpolated_line2, [parameters.channel1, parameters.channel2])
    ax1.axes.yaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    
    #limits
    ax1.set_xlim(0,20)    
    ax1.tick_params(axis='y', labelcolor=parameters.color1)
    ax2.tick_params(axis='y', labelcolor=parameters.color2)
    plt.xticks(range(0, len(line1), 2),range(-int(len(line1)*parameters.pixel_size/2), int(len(line1)*parameters.pixel_size/2),parameters.pixel_size*2))
    
    #Peak difference calculation and display
    ax1.text(0.3, np.max(f1n(xnew))*0.99, f'Peak Difference: {((np.argmax(f1n(xnew)) -np.argmax(f2n(xnew))) * parameters.pixel_size / ip_pres):.1f}nm')

    #Display
    plt.show()    