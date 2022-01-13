import cv2
import numpy as np
import glob
from dataset import Dataset
import shutil
import tensorflow as tf 

src = glob.glob("data/01/*.txt")
dst = open("data/object_detection_dataset/annotation.txt" , "w")


for txt_file in src:
    img_file = txt_file[:-4] + ".jpg"
    print(img_file)
    img = cv2.imread(img_file)
    annotation = open(txt_file).readline()
    annotation = annotation.split(",")[:8]
    annotation = [float(i) for i in annotation]
    annotation = np.array(annotation).reshape(4,2)
    tl = np.min(annotation , 0)
    br = np.max(annotation , 0)
    dst.write("{},{},{},{},{} \n".format(img_file.split("/")[-1] , tl[0] , tl[1] , br[0] , br[1]))
    shutil.copyfile(img_file , "data/object_detection_dataset/" + img_file.split("/")[-1])
    h,w = img.shape[:2]
    tl = [int(tl[0] * w)  , int(tl[1] * h)]
    br = [int(br[0] * w)  , int(br[1] * h)]
    img = cv2.rectangle(img , tl, br , (255,255,0) , 5)

    cv2.imwrite("test.jpg", img)