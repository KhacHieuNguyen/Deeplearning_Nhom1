import tensorflow as tf
import numpy as np
import argparse
import pdb
import cv2
import pdb
from model import Model
from hparams import hparams


class FullSequenceModel(tf.keras.layers.Layer):
    def __init__(self, conv_layer, hparams):
        super(FullSequenceModel, self).__init__()
        self.conv_layer = conv_layer
        #self.decoder = decoder
        self.hparams = hparams

    def call(self, image):
        current_batch_size = 1
        image = image  / 255
        result = self.conv_layer(image, training=False)
        #result = []
        #predictions = self.conv_layer(batch_input)

        # h, w, d = image.shape
        # new_h = hparams.max_height
        # max_w = hparams.max_width
        # unpad_im = cv2.resize(im, (int(new_h*w/h), new_h), interpolation = cv2.INTER_AREA)
        # if unpad_im.shape[1] > max_w:   
        #     pad_im = cv2.resize(im, (max_w, new_h), interpolation = cv2.INTER_AREA)
        # else:
        #     pad_im = cv2.copyMakeBorder(unpad_im,0,0,0,max_w-int(new_h*w/h),cv2.BORDER_CONSTANT,value=[0,0,0])
        #     h1, w1, d1 = unpad_im.shape 
        #     h2, w2, d2 = pad_im.shape
        #result = model.inference(image)
        

        # for t in range(self.hparams.max_char_length):
        #     predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, conv_out)
        #     #for models without softmax at training time
        #     # predictions = tf.nn.softmax(predictions)
        #     result.append(predictions)
        #     dec_input = tf.argmax(predictions, axis=1)
        #     dec_input = tf.cast(tf.expand_dims(dec_input, axis=1), tf.float32)
        
        return result



def main(args):
    ####### load pretrain model ####################################
    # model = Model()
    # model.create_model_inference()
    # model.checkpoint.restore(args.ckpt)
    model = Model()
    model.create_model()
    # model.load_model()
    model.load_weight_from_path(args.ckpt)
    # inference
    im = cv2.imread(args.image_path) 
    h, w, d = im.shape
    new_h = hparams.max_height
    max_w = hparams.max_width

    unpad_im = cv2.resize(im, (int(new_h*w/h), new_h), interpolation = cv2.INTER_AREA)
    if unpad_im.shape[1] > max_w:   
        pad_im = cv2.resize(im, (max_w, new_h), interpolation = cv2.INTER_AREA)
    else:
        pad_im = cv2.copyMakeBorder(unpad_im,0,0,0,max_w-int(new_h*w/h),cv2.BORDER_CONSTANT,value=[0,0,0])
        h1, w1, d1 = unpad_im.shape 
        h2, w2, d2 = pad_im.shape
        
    result = model.inference(pad_im)
    print (result)
    result = result.numpy()
    result [(result<0)] = 0
    result [(result>1)] = 1.0

    result = list (result[0,:])
    if unpad_im.shape [1] <= max_w:
        result = [p*w2/w1 if i < 4 else p*h2/h1 for i, p in enumerate(result)]
        

    
    #list_pts = [int(float(p) * hparams.max_width) if i < hparams.num_point else int(float(p) * hparams.max_height) for i, p in enumerate(result)]
    #list_pts = [int(float(p) * w) if i < hparams.num_point else int(float(p) * h) for i, p in enumerate(result)]
    #pts = [(list_pts[i], list_pts[i + hparams.num_point]) for i in range(hparams.num_point)]
    

    image_var = tf.Variable(initial_value=np.array([pad_im]), trainable=True, dtype=tf.uint8)


    ####### test full sequence model ####################################
    fsm = FullSequenceModel(model.conv_layer,  hparams)
    print('image_var shape: ',image_var.shape)
    # result = fsm(image_var)
    # print (result)
    



    ###### export to saved model ####################################
    model_input = tf.keras.layers.InputLayer(input_shape=(hparams.max_height, hparams.max_width, 3), batch_size=1, dtype = tf.uint8)
    print(model_input)
    model_input = tf.cast(model_input.output, dtype=tf.float32)

    test_model = tf.keras.models.Sequential([model_input, fsm])
    result = test_model(image_var, training=False)
    # result = tf.strings.reduce_join(result).numpy().decode('utf-8')
    # print(result.shape)
    print(result)
    
    test_model.save(args.sm)
    print('save model to saved_model')

def load_savedmodel(img_path, model_path):
    model = tf.saved_model.load(model_path)

    im = cv2.imread(img_path) 
    h, w, d = im.shape
    new_h = 400
    max_w = 400
    num_point = 4

    unpad_im = cv2.resize(im, (int(new_h*w/h), new_h), interpolation = cv2.INTER_AREA)
    if unpad_im.shape[1] > max_w:   
        pad_im = cv2.resize(im, (max_w, new_h), interpolation = cv2.INTER_AREA)
    else:
        pad_im = cv2.copyMakeBorder(unpad_im,0,0,0,max_w-int(new_h*w/h),cv2.BORDER_CONSTANT,value=[0,0,0])
        h1, w1, d1 = unpad_im.shape 
        h2, w2, d2 = pad_im.shape
    image_var = tf.Variable(initial_value=np.array([pad_im]), trainable=True, dtype=float)

    result = model (image_var, training=False)
    print (result)

    result = result.numpy()
    result [(result<0)] = 0
    result [(result>1)] = 1.0

    result = list (result[0,:])
    if unpad_im.shape [1] <= max_w:
        result = [p*w2/w1 if i < 4 else p*h2/h1 for i, p in enumerate(result)]
    list_pts = [int(float(p) * w) if i < num_point else int(float(p) * h) for i, p in enumerate(result)]
    pts = [(list_pts[i], list_pts[i + num_point]) for i in range(num_point)]
    for p in pts:
        cv2.circle(im, p, 4, (0, 0, 255), -1)
    cv2.imwrite('result.jpg', im)
    return pts



# pts = load_savedmodel ('test.jpg', 'savedmodel')
# print (pts)
# abc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='test.jpg', help='path to inference image')
    parser.add_argument('--ckpt', default='/data1/users/tuanb/OCR_DOI/training_checkpoints/hparams_1/train/ckpt-34', help='path to trained ckpt')
    parser.add_argument('--sm', default='savedmodel_1803', help='path to saved_model')
    args = parser.parse_args()
    main(args)
