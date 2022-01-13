import glob 
import shutil
import numpy as np 

training_rate = 0.8
val_rate = 0.9
test_rate = 1

src = glob.glob("data/01/*.txt")

num_sample = len(src)
permutation = np.random.permutation(num_sample)

for i in range(int(training_rate * num_sample)):
    index = permutation[i]
    txt_file = src[index]
    img_file = txt_file[:-4] + ".jpg"
    dst_img = "data/train/" + img_file.split("/")[-1]
    dst_txt = dst_img[:-4] + ".txt"
    shutil.copy(img_file , dst_img)
    shutil.copy(txt_file , dst_txt)

for i in range(int(training_rate * num_sample) , int(val_rate * num_sample) ):
    index = permutation[i]
    txt_file = src[index]
    img_file = txt_file[:-4] + ".jpg"
    dst_img = "data/val/" + img_file.split("/")[-1]
    dst_txt = dst_img[:-4] + ".txt"
    shutil.copy(img_file , dst_img)
    shutil.copy(txt_file , dst_txt)

for i in range(int(val_rate * num_sample) , num_sample):
    index = permutation[i]
    txt_file = src[index]
    img_file = txt_file[:-4] + ".jpg"
    dst_img = "data/test/" + img_file.split("/")[-1]
    dst_txt = dst_img[:-4] + ".txt"
    shutil.copy(img_file , dst_img)
    shutil.copy(txt_file , dst_txt)