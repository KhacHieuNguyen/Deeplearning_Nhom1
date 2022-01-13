import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras.applications.efficientnet import EfficientNetB1
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from hparams import hparams

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ConvLayer, self).__init__()
        base_model = EfficientNetB1(include_top=False, weights='imagenet')
        base_model_layers = [layer.name for layer in base_model.layers]
        dense_layer = tf.keras.layers.Dense (1024, activation = 'relu') (base_model.get_layer(hparams.end_point).output)
        dense_layer = tf.keras.layers.Dense (512, activation = 'relu') (dense_layer)
        conv_tower_output = tf.keras.layers.Dense (hparams.num_point*2, activation='linear') (dense_layer)
        
        self.conv_model = tf.keras.models.Model(inputs=base_model.input, outputs=conv_tower_output)
        self.conv_model.summary()
    
    def call(self,target):
        return self.conv_model(target)
