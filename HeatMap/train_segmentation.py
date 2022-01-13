from calendar import c
import tensorflow as tf
import numpy as np
import glob
import cv2
import os
from segmentation import Deeplabv3, deeplab
from tensorflow.keras.callbacks import ModelCheckpoint
from model import Model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# IoU metric

def main():
    ### create model
    model = Model()
    model.create_model()
    #model.load_model()
    ### train new model
    model.train()

if __name__ == '__main__':
    main()


