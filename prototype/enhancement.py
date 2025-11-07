import cv2
import numpy as np
import random

CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (16, 32)

def apply_enhancement(image):

    if image is None:
        return None 

    # Random Gamma Correction
    random_gamma = random.uniform(0.5, 1.0)
    gamma_table = np.array([((i / 255.0) ** random_gamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected_image = cv2.LUT(image, gamma_table)

    #BGR to YCrCb Conversion

    ycrcb_image = cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2YCrCb)

    # Apply CLAHE on the Luminance (Y) channel

    y, cr, cb = cv2.split(ycrcb_image)
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    clahe_y = clahe.apply(y)
    clahe_ycrcb = cv2.merge([clahe_y, cr, cb])

    # Convert back to BGR
    final_image = cv2.cvtColor(clahe_ycrcb, cv2.COLOR_YCrCb2BGR)

    return final_image