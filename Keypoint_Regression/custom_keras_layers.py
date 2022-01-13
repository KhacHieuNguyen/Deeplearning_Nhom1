import efficientnet.tfkeras as efn
# import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import collections
import logging
import pdb

from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet 

logging.basicConfig(level=logging.DEBUG)




class ConvBaseLayer(tf.keras.layers.Layer):
    def __init__(self, hparams):
        super(ConvBaseLayer, self).__init__()
        self.hparams = hparams
        if hparams.base_model_name == 'InceptionV3':
            base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'InceptionResNetV2':
            base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB0':
            base_model = efn.EfficientNetB0(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
            dense_layer = tf.keras.layers.Dense (512, activation = 'relu') (base_model.get_layer(hparams.end_point).output)
        elif hparams.base_model_name == 'EfficientNetB1':
            base_model = efn.EfficientNetB1(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]

        elif hparams.base_model_name == 'EfficientNetB2':
            base_model = efn.EfficientNetB2(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB3':
            base_model = efn.EfficientNetB3(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB4':
            base_model = efn.EfficientNetB4(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB5':
            base_model = efn.EfficientNetB5(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'EfficientNetB6':
            base_model = efn.EfficientNetB6(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
        elif hparams.base_model_name == 'mobinet':
            base_model = MobileNet(include_top=False, weights='imagenet')
            base_model_layers = [layer.name for layer in base_model.layers]
            dense_layer = tf.keras.layers.GlobalAveragePooling2D () (base_model.get_layer(hparams.end_point).output)
            #dense_layer = Dense (512, activation = 'relu') (dense_layer)
        #dense_layer = Dense (1024, activation = 'relu') (dense_layer)

        assert hparams.end_point in base_model_layers, "no {} layer in {}".format(hparams.end_point, hparams.base_model_name)
        
        
        dense_layer = tf.keras.layers.Dense (1024, activation = 'relu') (base_model.get_layer(hparams.end_point).output)
        dense_layer = tf.keras.layers.Dense (512, activation = 'relu') (dense_layer)
        #dense_layer = Dense (128, activation = 'relu') (dense_layer)
        conv_tower_output = tf.keras.layers.Dense (hparams.num_point*2, activation='linear') (dense_layer)
        
        self.conv_model = tf.keras.models.Model(inputs=base_model.input, outputs=conv_tower_output)
        print (self.conv_model.summary())


        

    def call(self, inputs):
        conv_out  = self.conv_model(inputs)
        
        return conv_out


