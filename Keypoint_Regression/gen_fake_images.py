import numpy as np
import cv2
import os
from time import time

"""
Background is resized to (1600x1600)
split into 4 pieces, 4 coordinate of (tl, tr, br, bl) will be in each pieces
"""
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # top-right have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect  # tl, tr, br, bl order

def get_random_coords(w_bg, h_bg):
    # return np.array([[100, 100], [700, 200], [800, 800], [550, 800]], dtype='float32')
    part_w = w_bg // 10
    part_h = h_bg // 10
    tl = (np.random.randint(0, 4*part_w), np.random.randint(0, 4*part_h))
    tr = (np.random.randint(6*part_w, w_bg), np.random.randint(0, 4*part_h))
    br = (np.random.randint(6*part_w, w_bg), np.random.randint(6*part_h, h_bg))
    bl = (np.random.randint(0, 4*part_w), np.random.randint(6*part_h, h_bg))
    coords = np.array([tl, tr, br, bl])

    return coords

def paste(src, dst, bg_size):
    dst_coords = get_random_coords(bg_size[0], bg_size[1])
    h_s, w_s = src.shape[:2] # 869, 1260
    h_d, w_d = dst.shape[:2]
    src_coords = np.array([
        [0, 0], [w_s-1, 0], [w_s-1 , h_s-1], [0, h_s-1]], dtype="float32")
    
    trans, _ = cv2.findHomography(src_coords, dst_coords)

    dst_map = cv2.warpPerspective(src, trans, (h_d, w_d))
    
    mask = np.array(dst_map > 0, dtype=np.uint8)
    res = dst*(1-mask) + dst_map*mask
   
    return res, dst_coords.astype('int')

def draw_dot(image, points: list):
    for i, p in enumerate(points):
        color = (0, 0, 255)
        cv2.putText(image, str(i), tuple(p), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        cv2.circle(image, tuple(p), 4, color, -1)
    return image

def write_label(path, coords):
    '''
    coords: 1-D array with format x1, y1, x2, y2, ...
    '''
    with open(path, 'w') as f:
        f.write('%f,%f,%f,%f,%f,%f,%f,%f,'%tuple(coords))

if __name__ == '__main__':
    
    rois_path = 'ROI'
    save_path = 'gen_nn'
    bg_path = 'bg_img_2'
    # bg_path = '/home/local/VNG/passport/data/training_set_son_idcard_padding/training_set/clear'
    bgs_p = [os.path.join(bg_path, name) for name in os.listdir(bg_path)]
    bg_size = (1600, 1600) #w, h
    nums_bg = len(bgs_p)
    # bg = cv2.imread(r"bg\cream_15.jpg")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for name in os.listdir(rois_path):

        print('use image: ', name)
        s = time()
        fp = os.path.join(rois_path, name)
        img = cv2.imread(fp)
        if img is None:
            continue
       
        idx = np.random.randint(nums_bg)
        bg = cv2.imread(bgs_p[idx])
        if bg is None:
            continue
        bg = cv2.resize(bg, bg_size)
        t1 = time()
        res, coords = paste(img, bg.copy(), bg_size)
        # res = draw_dot(res, coords)
        t2 = time()
        uid = np.random.randint(65536)
        sp = os.path.join(save_path, name.split('.')[0] + '_%d.jpg'%uid)
        cv2.imwrite(sp, res)
        coords_std = [c / bg_size[0] if i%2==0 else c/bg_size[1] for i, c in enumerate(coords.ravel())]
        write_label(sp.replace('.jpg', '.txt'), coords_std)
        # cv2.namedWindow('res', cv2.WINDOW_NORMAL)
        # cv2.imshow('res', res)
        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     exit()