import numpy as np
import cv2
import math


def colorPick(rawImage):

	image = cv2.imread(rawImage)[:,:,::-1]
	height = math.ceil(image.shape[1] * 0.1)
	width = math.ceil(image.shape[0] * 0.1)
	mid = (math.ceil(image.shape[0]/2), math.ceil(image.shape[1]/2)) 

	croppedImage = image[mid[0]-width:mid[0]+width, mid[1]-height:mid[1]+height]

	rvalue = np.mean([np.mean(pixel[0]) for pixel in croppedImage[0]])
    bvalue = np.mean([np.mean(pixel[2]) for pixel in croppedImage[0]])

    if rvalue > bvalue:
    	return 'red'
    else:
    	return 'blue'