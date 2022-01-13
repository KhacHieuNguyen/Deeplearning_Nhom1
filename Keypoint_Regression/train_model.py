import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Conv2D, \
                    Dropout, Flatten, MaxPool2D
from tensorflow.keras.losses import mse
from tensorflow.keras.activations import sigmoid,relu
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import cv2
import glob


from tensorflow.python.keras.layers.convolutional import Conv2DTranspose




def load_data(folder, img_shape = (128,128)):
    X =[]
    y = []
    annotations = glob.glob(folder + "/*.txt")
    for annotation in annotations:
        # print("loading {}".format(annotation))
        img_path = annotation[:-4] + ".jpg"
        image = cv2.imread(img_path)
        image = cv2.resize(image , img_shape[:2])
        X.append(image)
        label = open(annotation).readline()
        label = label.split(",")[:8]
        label = [float(i) for i in label]
        label = np.array(label)
        y.append(label)

    return (np.array(X) , np.array(y))


def residual_block(input, filters = 128 , kernel_size = (3,3)):
    conv1 = Conv2D(filters, kernel_size = kernel_size,strides = (2,2),activation=relu, padding="same")(input)
    conv2 = Conv2D(filters ,kernel_size=kernel_size,activation=relu, padding = "same" )(conv1) 
    maxpool = MaxPool2D(strides=(2,2) , padding = "same")(input)
    return conv2 + maxpool


#config

IMG_SHAPE = (256,256,3)

model_dir = "model/"

BS = 32
EPOCHS = 200
#build model

input = tf.keras.layers.Input(IMG_SHAPE)
conv1 = Conv2D(filters = 64,kernel_size=(11,11),activation=relu)(input)

maxpool1 = MaxPool2D(strides=(2,2))(conv1)
res1 = residual_block(maxpool1 , filters = 64)
res2 = residual_block(res1, filters = 64)
conv2 = Conv2D(filters = 128 , kernel_size = (3,3) ,strides =  (2,2) ,activation= relu)(res2)
flat = Flatten()(conv2)

fc1 = Dense(512 , relu )(flat)

output = Dense(8)(fc1)

model = tf.keras.models.Model(input , output)

model.summary()



#train model
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    1e-3, 1000, end_learning_rate=1e-5)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer= opt , loss = mse , metrics = [mse])

(X,y) = load_data("annotations/train", IMG_SHAPE[:2])
X = X * 1./255

filepath = model_dir+"model-{epoch:02d}-{val_loss:.3f}.h5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False,)
model.fit(X , y ,
            batch_size=BS,
            validation_split=0.1,
            epochs =EPOCHS,
            callbacks = [checkpoint])

model.save("model.h5")

