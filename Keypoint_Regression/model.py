from custom_keras_layers import ConvBaseLayer
from datetime import datetime
from dataset import Dataset
from hparams import hparams
from tqdm import tqdm

# from custom_model import ConvLayer
import tensorflow as tf

import numpy as np
import logging
# logging.basicConfig(level=logging.DEBUG)
import pdb
import os
# import tensorflow.compat.v2 as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#from object_detection import model_lib_v2
# from keras.applications import mobilenet
import math

class Model(object):
    def __init__(self):
        self.best_val_acc = 100.0
        self.ld = 0.2
        # dataset
        self.train_dataset = Dataset(hparams, hparams.train_record_path)
        self.valid_dataset = Dataset(hparams, hparams.valid_record_path)
        self.train_dataset.load_tfrecord()
        self.valid_dataset.load_tfrecord()

    def loss_function(self, real, pred):
        loss_ = self.loss_object(real, pred)
        loss_1 = tf.reduce_mean(loss_)
        loss_ = self.loss_object(self.quadrant_angle (real), self.quadrant_angle(pred))
        loss_2 = tf.reduce_mean(loss_)
        loss = loss_1 + self.ld * loss_2
        return loss

    def quadrant_angle (self, labels):
    	x_cord = [labels[:,0] * hparams.max_width, labels[:,2] * hparams.max_width, labels[:,4] * hparams.max_width, labels[:,6] * hparams.max_width]
    	y_cord = [labels[:,1] * hparams.max_height, labels[:,3] * hparams.max_height, labels[:,5] * hparams.max_height, labels[:,7] * hparams.max_height]
        
    	angle = np.arctan2(y_cord, x_cord) * 180.0 / np.pi
    	#normalize to range (0,360)
    	mask = (angle < 0)
    	angle[mask] += 360
    	mask = (angle > 360)
    	angle[mask] -= 360 
    	return angle /360.0
    # def quadrant_angle (self, labels):
    #     x_cord = labels[:,:hparams.num_point] * hparams.max_width
    #     y_cord = labels[:,hparams.num_point:] * hparams.max_height
    #     angle = np.arctan2(y_cord, x_cord) * 180.0 / np.pi
    #     #normalize to range (0,360)
    #     mask = (angle < 0)
    #     angle[mask] += 360
    #     mask = (angle > 360)
    #     angle[mask] -= 360 
    #     return angle /360.0

    # def angle(self, labels):
    # 	arctan1 = [math.atan2 (label[1] - label[0], label[5] - label [4]) for label in lables]
    # 	arctan2 = [math.atan2 (label[2] - label[1], label[6] - label [5]) for label in labels]
    # 	arctan3 = [math.atan2 (label[3] - label[2], label[7] - label [6]) for label in labels]
    # 	arctan4 = [math.atan2 (label[0] - label[3], label[4] - label [7]) for label in labels]



    # 	theta_12 = 180 - (180 * (arctan1 - arctan2) / math.pi)
    # 	theta_23 = 180 - (180 * (arctan2 - arctan3) / math.pi)
    # 	theta_34 = 180 - (180 * (arctan3 - arctan4) / math.pi)
    # 	theta_41 = 180 - (180 * (arctan4 - arctan1) / math.pi)

    # 	thetas = [theta_12, theta_23, theta_34, theta_41]
    # 	theta_norm = []
    # 	for theta in thetas:
    # 		if theta > 360:
    # 			theta_norm.append (theta - 360)
    # 		elif theta < 0:
    # 			theta_norm.append (theta + 360)
    # 		else:
    # 			theta_norm.append (theta)
    # 	return theta_norm



    def create_model(self):
        ### create model
        self.conv_layer = ConvBaseLayer(hparams)
        ### define training ops and params
        #self.conv_layer = ConvLayer()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams.lr)
        self.loss_object = tf.keras.losses.MeanSquaredError ()#from_logits=True, reduction='none'

        self.last_epoch = 0
        self.train_summary_writer = tf.summary.create_file_writer(hparams.save_path + '/logs/train')
        self.valid_summary_writer = tf.summary.create_file_writer(hparams.save_path + '/logs/valid')
        self.checkpoint_dir = os.path.join(hparams.save_path, 'train')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.conv_layer)

    def load_model(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        print(latest)
        if latest != None:
            logging.info('load model from {}'.format(latest))
            self.last_epoch = int(latest.split('-')[-1])
            self.checkpoint.restore(latest)

    def load_weight_from_path(self,path):
        logging.info('load model from {}'.format(path))
        self.last_epoch = int(path.split('-')[-1])
        self.checkpoint.restore(path)
    
    
    def train_step(self, batch_input, batch_target):
        loss = 0
        current_batch_size = batch_input.shape[0]
        with tf.GradientTape() as tape:
            predictions = self.conv_layer(batch_input)
            loss += self.loss_function(batch_target, predictions)

        variables = self.conv_layer.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return (loss / int(batch_target.shape[1]))

    def evaluate(self, batch_input, batch_target):
        # current_batch_size = batch_input.shape[0]
        predictions = self.conv_layer(batch_input)
        mse_loss = self.loss_function(predictions, batch_target)
        return mse_loss

    def inference(self, image):
        # current_batch_size = 1
        # image = image * 1./255
        result = self.conv_layer(np.array([image]))         
        return result

    def train(self):
        self.load_model()
        for epoch in range(self.last_epoch, hparams.max_epochs):
            start = datetime.now()
            total_loss = 0
            # train each batch in dataset
            sum_samples = 0

            for batch, (batch_input, batch_target) in enumerate(self.train_dataset.dataset):
                batch_loss = self.train_step(batch_input, batch_target)
                total_loss += batch_loss
                
                if batch % 1 == 0:
                    print('Epoch {} Batch {} Loss {:.8f}'.format(epoch + 1, batch, batch_loss.numpy()))

            
            
            sum_samples = 0
            total_loss_val = 0
           

            for batch, (batch_input, batch_target) in enumerate(self.valid_dataset.dataset):
                batch_loss = self.evaluate(batch_input, batch_target)
                total_loss_val+= batch_loss
                sum_samples  += 1
            valid_acc  = total_loss_val/sum_samples

            # save checkpoint
            if hparams.save_best:
                if self.best_val_acc > valid_acc:
                    self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                    self.best_val_acc = valid_acc
            else:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            # write log
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', total_loss, step=epoch)
                tf.summary.scalar('Loss train', total_loss, step=epoch)
                
            with self.valid_summary_writer.as_default():
                tf.summary.scalar('loss test', valid_acc, step=epoch)
                

            # log traing result of each epoch
            logging.info('Epoch {} Loss {:.8f}'.format(epoch + 1, total_loss / batch))
            logging.info('Accuracy on train set: {:.6f}'.format(total_loss))
            logging.info('Accuracy on valid set: {:.6f}'.format(valid_acc))
            logging.info('Time taken for 1 epoch {} sec\n'.format(datetime.now() - start))
