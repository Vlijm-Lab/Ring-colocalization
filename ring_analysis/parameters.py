# Parameters can be set here. All parameters will be stored in the Parameters object which is used by the peak_difference.ipynb 

#Pixel size in nm
pixel_size = 17

#Directory containing the raw data (.tiff files) in the form of (2, ..., ...), i.e. a dual stack image.
img_dir = 'data'

#Set to "True" if the first channel is the dominant channel, and set to "False" if the second channel is the dominant channel (from which the peaks are used to center the line profiles on)
top_layer_dominant = False

#Channel names, give a representative name for both channels. 
channel1 = "CdvB"
channel2 = "CdvB1"

#Set to "True" for displaying raw data with corresponding colormaps
display_images = True

#Set to "True" for displaying the determination of the peak position in the dominant channel.
display_diameter_calculation = True


#Colormaps of first and second channel, respectively
color1 = "magenta"
color2 = "green"

#parameters which determine the number of line profiles obtained per image
number_rotations = 180
rotation_step = 1

class Parameters:
    '''
    Object containing parameters for the peak to peak analysis.
   
    Parameters
    ----------
        pixel_size : float
             pixel size in nm
        img_dir : str
            path to the directory containing the dual layer .tiff images
        top_layer_dominant : bool
            set which layer is dominant and therefore used for peak centering of line profiles
        channel1 : str
            name of first layer of image stack
        channel1 : str
            name of second layer of image stack
        color1 : str
            colormap for first layer of the image stack
        color2 : str
            colormap for second layer of the image stack
        display_images : bool
            use to display raw images.
        display_diameter_calculation : bool
            use to display diameter calculations.
        number_rotations : int
            number of rotations used per image to obtain line profiles
        rotation_step : int
            size of the increase in rotation which is used every itteration to rotate the image
    '''
    
    def __init__(self, pixel_size, img_dir, top_layer_dominant, channel1, channel2, color1, color2, display_images, display_diameter_calculation, number_rotations, rotation_step):
        self.pixel_size = pixel_size
        self.img_dir = img_dir
        self.top_layer_dominant = top_layer_dominant
        self.channel1 = channel1
        self.channel2 = channel2
        self.color1 = color1
        self.color2 = color2
        self.display_images = display_images
        self.display_diameter_calculation = display_diameter_calculation
        self.number_rotations = number_rotations
        self.rotation_step = rotation_step
        
def load_parameters():
    '''
    Load all set parameters defined in ring_analysis/parameters.py in a Parameters object.

            Parameters:
                    None

            Returns:
                    parameters (Parameters): Class containing all parameters or the peak to peak analysis.
    '''

    parameters = Parameters(pixel_size, img_dir, top_layer_dominant, channel1, channel2, color1, color2, display_images, display_diameter_calculation, number_rotations, rotation_step)
    
    return parameters