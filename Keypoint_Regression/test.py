import numpy as np 
import tensorflow as tf
from model import Model
from process_image import get_RoI, test_image
import glob
import cv2
import os
from hparams import hparams
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_path = "model_512Unit/train/ckpt-479"
dst_dir = "data/479/"

detector = Model()
detector.create_model()
detector.load_weight_from_path(model_path)


def widen_annotation(annotation, factor = 0.06):
    annotation = np.array(annotation).reshape((4,2))
    tl = annotation[0]
    tr = annotation[1]
    br = annotation[2]
    bl = annotation[3]

    center = (tl + br) / 2
    tl = center + (tl - center) * (1 + factor / 2)
    br = center + (br - center) * (1 + factor / 2)
    center = (tr + bl) / 2
    tr = center + (tr - center) * (1 + factor / 2)
    bl = center + (bl - center) * (1 + factor / 2)
    return annotation


def main():
    folder = glob.glob("data/test/*")
    for (i,file) in enumerate(folder):
        print(str(i),";,",file)
        if ".txt" in file :
            continue
        image = cv2.imread(file)
        if image is None: continue
        h,w,d = image.shape
        ps = max(h, w) // 10
        # pad_im = cv2.copyMakeBorder(image, ps, ps, ps, ps, cv2.BORDER_REPLICATE)
        pad_im_resize = cv2.resize(image, (400,614), interpolation = cv2.INTER_AREA)

 
        
        result = detector.inference(pad_im_resize * 1./255 )
        result = result.numpy() # get output numpy array shape = (8)
        # result = widen_annotation(result[0])
        result [(result<0)] = 0
        result [(result>1)] = 1.0
        dst_path = dst_dir + file.split("/")[-1]
        # pad_im = test_image(pad_im , result[0] )
        out = get_RoI(pad_im_resize, result.reshape((4,2)))
        cv2.imwrite(dst_path,np.transpose(out,(1,0,2)))
if __name__ == "__main__":
    # image = cv2.imread("_test/11896212_843468669082574_5751569227034550861_n_24557.jpg")
    # result = detector.inference(image * 1./255)
    # dst_path = "result.jpg"
    # image = test_image(image , result[0])

    # cv2.imwrite(dst_path,image)
    main()
