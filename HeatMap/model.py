import tensorflow as tf
import numpy as no
import os
import logging
import numpy as np


from datetime import datetime
from segmentation import Deeplabv3
from dataset import Dataset 
from hparams import hparams


class Model(object):
    def __init__(self):
        self.best_val_acc = 100.0
        self.ld = 0.2
        # dataset
        self.train_dataset = Dataset(hparams, hparams.train_record_path)
        self.valid_dataset = Dataset(hparams, hparams.valid_record_path)
        self.train_dataset.load_tfrecord()
        self.valid_dataset.load_tfrecord()
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.evaluation_metric = tf.keras.metrics.MeanIoU(num_classes=2)


    def loss_function(self, real, pred):
        
        return self.loss(real , pred)

    def create_model(self):
        self.conv_layer =  Deeplabv3(input_shape = (800,800,3))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams.lr)
        self.last_epoch = 0
        self.train_summary_writer = tf.summary.create_file_writer(hparams.save_path + '/logs/train')
        self.valid_summary_writer = tf.summary.create_file_writer(hparams.save_path + '/logs/valid')
        self.checkpoint_dir = os.path.join(hparams.save_path, 'train')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,encoder=self.conv_layer)
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
        val_loss = self.loss_function(predictions , batch_target)

        prediction = tf.cast( predictions >  hparams.threshold, tf.bool)
        target = tf.cast(batch_target > 0 , tf.bool)
        Inter = tf.cast(tf.math.logical_and(target , prediction) , tf.float64 )
        Union = tf.cast(tf.math.logical_or(target , prediction) , tf.float64)

        return tf.reduce_mean(Inter) / (tf.reduce_mean(Union) + 1e-8) , val_loss
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
            total_IoU= 0
            total_val_loss = 0
           

            for batch, (batch_input, batch_target) in enumerate(self.valid_dataset.dataset):
                batch_IoU , val_loss = self.evaluate(batch_input, batch_target)
                total_IoU+= batch_IoU
                total_val_loss += val_loss
                sum_samples  += 1
            valid_acc  = total_IoU / sum_samples
            mean_val_loss = total_val_loss / sum_samples 
            # save checkpoint
            if hparams.save_best:
                if self.best_val_acc > valid_acc:
                    self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                    self.best_val_acc = valid_acc
            else:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            # write log
            with self.train_summary_writer.as_default():
                tf.summary.scalar('train loss', total_loss, step=epoch)
                
                
            with self.valid_summary_writer.as_default():
                tf.summary.scalar('val_IoU', valid_acc, step=epoch)
                tf.summary.scalar('val loss ', mean_val_loss, step=epoch)

            # log traing result of each epoch
            print('Epoch {} Loss {:.8f}'.format(epoch + 1, total_loss / batch))
            print('Total loss on train set: {:.10f}'.format(total_loss))
            print('Accuracy on valid set: {:.10f}'.format(valid_acc))
            print('Time taken for 1 epoch {} sec\n'.format(datetime.now() - start))