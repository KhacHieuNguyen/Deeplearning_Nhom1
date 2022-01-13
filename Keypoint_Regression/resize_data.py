import argparse
import pdb
import cv2
import os
import glob
import numpy as np


def main(args):
	data = os.listdir(args.raw_data)
	image_path = list (glob.iglob(os.path.join (args.raw_data, "*.jpg")))
	label_path = list (glob.iglob(os.path.join (args.raw_data, "*.txt")))
	
	max_w = 400
	new_h = 400

	for i, p_label in enumerate(label_path):
		image = p_label[0:-4] + '.jpg'
		image_name = image.split('/')[-1]
		save_path_img = os.path.join(args.pad_data, image_name)
		save_path_label = save_path_img [:-4]+ '.txt'
		
		if not os.path.isfile (image):
			image = p_label[0:-4] + '.jpeg'
			image_name = image.split('/')[-1]
			save_path_img = os.path.join(args.pad_data, image_name)
			save_path_label = save_path_img [:-5]+ '.txt'
			
		print(i, image)
		
		pts =[]
		with open(p_label, 'r') as f:
			label = f.read().split(',')
			pts = label[0:-1]
			pts = [float(p) for p in pts]

		im = cv2.imread(image)
		if im is None:
			continue
		h, w, d = im.shape
		unpad_im = cv2.resize(im, (int(new_h*w/h), new_h), interpolation = cv2.INTER_AREA)
		if unpad_im.shape[1] > max_w:	
			pad_im = cv2.resize(im, (max_w, new_h), interpolation = cv2.INTER_AREA)
			new_pts = pts
		else:
			pad_im = cv2.copyMakeBorder(unpad_im,0,0,0,max_w-int(new_h*w/h),cv2.BORDER_CONSTANT,value=[0,0,0])
			h1, w1, d1 = unpad_im.shape 
			h2, w2, d2 = pad_im.shape
			# for i in range (0,8):
			# 	if i%2 == 0:
			# 		new_pts.append(pts [i]*w1/w2)
			# 	else:
			# 		new_pts.append(pts [i]*h1/h2)
			new_pts = [p*w1/w2 if i % 2 == 0 else p*h1/h2 for i, p in enumerate(pts)]
				
		cv2.imwrite(save_path_img, pad_im)
		with open(save_path_label, 'w') as f:
			for p in new_pts:
				f.write('%f,' % p)

def show(img_path):
	pts = []
	with open(img_path.split('.')[0]+'.txt', 'r') as f:
		label = f.read().split(',')
		pts = label[0:-1]
		pts = [float(p) for p in pts]
	nums = len(pts)//2
	img = cv2.imread(img_path)
	h, w = img.shape[:2]
	print(h, w)
	pts = [int(float(p) * w) if i %2 == 0 else int(float(p) * h) for i, p in enumerate(pts)]
	pt = [(pts[i], pts[i + 1]) for i in range(0,8,2)]
	for p in pt:
		cv2.circle(img, p, 4, (0, 0, 255), -1)
	cv2.imshow('label', img)
	key = cv2.waitKey(0)

def read_label(fp, lp):
    img = cv2.imread(fp)
    h, w = img.shape[:2]
    with open(lp, 'r') as f:
        raws = f.read()
    lps = [r for r in raws.split(',') if r != '']
    lps = np.array([float(p)*w if i%2 == 0 else float(p)*h for i, p in enumerate(lps)]).astype('int32')
    print (len (lps))
    # lps = np.array([ [lps[i], lps[i+1]] for i in range(0, 4, 2)])
   
    return lps

def show2(fp, lp):
    img = cv2.imread(fp)
    
    lps = read_label(fp, lp)
    for i in range(0,8,2):
        cv2.circle(img, (lps[i], lps [i+1]), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i/2) , (lps[i], lps [i+1]), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
    
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    img = cv2.resize(img, (800, 800))
    cv2.imshow('img', img)
    key = cv2.waitKey(0)
    # if key == ord('q'):
    #     exit(-1)	


# show2('../data/labeled/Passport/add_padding/pad_aug_data/835e6f21b6fcb250e38f3d9cd1d6e0a1._random_brightness.jpg', '../data/labeled/Passport/add_padding/pad_aug_data/835e6f21b6fcb250e38f3d9cd1d6e0a1._random_brightness.txt')
# abc

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--raw_data")
	parser.add_argument("--pad_data")
	args = parser.parse_args()
	main(args)

"""
	python3 resize_data.py --raw_data='../data/labeled/Passport/add_padding/train' --pad_data='../data/labeled/Passport/add_padding/pad_train'
	python3 resize_data.py --raw_data='../data/labeled/Passport/add_padding/test' --pad_data='../data/labeled/Passport/add_padding/pad_test'

	python3 resize_data.py --raw_data='/home/local/VNG/cloud156/passport/record/gen_data' --pad_data='/home/local/VNG/cloud156/passport/record/pad_aug_data'

	python3 resize_data.py --raw_data='gen_son' --pad_data='pad_aug_data'

"""
