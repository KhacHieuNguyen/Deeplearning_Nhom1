import tensorflow as tf 
import cv2
import numpy as np 
from process_image import test_image
model = tf.keras.models.load_model("model/model-76-0.016.h5")
model.summary()

image = cv2.imread("annotations/test/18839180_1524194007612321_4213770914914518787_n.jpg")
image_copy = cv2.resize(image, (256,256)) 

pred = model.predict(np.array([image_copy * 1./255]) )

image = test_image(image, pred)

cv2.imwrite("result.jpg",image)

