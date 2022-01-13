import cv2
from process_image import get_RoI
import glob 
import numpy as np
images = glob.glob("data/01/*.jpg") 
dst = "data/RoI/"

for image in images:
    try:
        print(image)
        RoI = get_RoI(image)
        dst_image = dst + image.split("/")[-1]
        cv2.imwrite(dst_image, np.transpose(RoI,(1,0,2)))
    except:
        # no annotation 
        continue