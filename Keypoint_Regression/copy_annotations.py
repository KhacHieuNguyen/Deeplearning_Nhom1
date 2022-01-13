import cv2
import numpy as np 
import os 
import glob
import shutil


dst = "annotations/"
folder = glob.glob("hieu_label/*.txt")
for txt_file in folder:
    txt = txt_file.split("/")[-1]
    filename = txt.split(".")[0]
    annotation_dst = dst + txt
    shutil.copyfile(src = txt_file , dst = annotation_dst)
    images = glob.glob("hieu_label/{}*".format(filename))
    for file in images:
        if ".txt" in file:
            continue
        image = cv2.imread(file)
        cv2.imwrite(dst + filename + ".jpg" , image)
