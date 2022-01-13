from fileinput import filename
import tensorflow as tf
import argparse
import cv2
import os
import glob
import pdb


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialize_example(image_encoded, image_format, image_width, image_height, mask):
    feature = {
        'image/encoded': _bytes_feature(image_encoded),
        'image/format': _bytes_feature(image_format),
        'image/width': _int64_feature(image_width),
        'image/height': _int64_feature(image_height),
        # 'image/class': _float_list_feature(image_class),
        #'image/unpadded_class': _int64_list_feature(unpadded_class),
        'image/mask': _bytes_feature(mask)
    }
  
    # Create a Features message using tf.train.Example.
  
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def main(args):

    save_path = args.out_path
    mask_files = list (glob.iglob(os.path.join (args.mask_path, "*.jpg")))
    image_files = [args.pad_path + file.split("/")[-1] for file in mask_files]

    cnt = 0
    error_files = []
    with tf.io.TFRecordWriter(save_path) as writer:
        for i, mask_file in enumerate(mask_files):
            filename = args.mask_path + mask_file.split("/")[-1]
            print(i, filename)
            try:
                #image/encoded
                # pdb.set_trace()
                # image_encoded = []
                with open(filename, "rb") as f:
                    image_encoded = f.read()
                cnt += 1
                # image/format
                ext = os.path.splitext (filename) [1]
                # print (ext)
                if ext == '.jpg':
                    image_format = "jpg".encode()
                if ext == '.jpeg':
                    image_format = "jpeg".encode()
                # image/orig_width
                image_height, image_width, _ = cv2.imread(filename).shape
                
                # image/class
                # with open(os.path.splitext (filename)[0] + '.txt', 'r') as f:
                #     image_class = f.read().split(',') [:8]
                #     image_class = [float(p) for p in image_class]
                #     print (image_class)
                # image/text
                # mask = cv2.imread(mask_file , cv2.CV_LOAD_IMAGE_GRAYSCALE)
                with open(mask_file, "rb") as f:
                    mask = f.read()
                # write to TFRecordFile
                example = serialize_example(image_encoded, image_format, image_width, image_height, mask)
                writer.write(example)
                print ("==========DONE=============")
            except Exception as e:
                print(e)
            
    print()
    print(cnt)
    for file in error_files:
        print(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pad_path", help = "directory contain images", default = "data/01/")
    parser.add_argument("--out_path", help = "output path", default = "data/data.train")
    parser.add_argument("--mask_path", help = "output path", default = "data/Segmentation_annotation/train/")
    args = parser.parse_args()
    main(args)

"""
python3 gen_record.py --pad_path data/01/ --mask_path data/Segmentation_annotation/test/ --out_path data/data.test 
  
   
"""
