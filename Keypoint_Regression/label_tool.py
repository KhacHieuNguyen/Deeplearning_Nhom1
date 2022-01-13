import os
import cv2
from sys import argv
import numpy as np

def click(event, x, y, flag, params):
    raw_coords, coords, img =  params['raw_coords'], params['coords'], params['img']
    h, w = img.shape[:2]
    if event == cv2.EVENT_LBUTTONDOWN:
        p = (x, y)
        raw_coords.append(p)
        coords.append((x / w, y / h))
        if len(raw_coords) > 0:
            cv2.line(params['img'], raw_coords[-1], p, (0, 255, 0), 2)
        cv2.circle(img, p, 4, (0, 0, 255), -1)
        cv2.putText(img, str(params['i']), p, cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        params['i']  += 1
        cv2.imshow('img', img)


def write_label(lp, cords):
    with open(lp, 'w') as f:
        X = [cord[0] for cord in cords]
        Y = [cord[1] for cord in cords]
        num_p = len(X) 
        assert num_p in [4], 'Expected 4 points, found: %d'%num_p

        for x, y in zip(X[:num_p], Y[:num_p]):
            f.write('%f,%f,'%(x, y))
        # f.write('\n')
        # for x, y in zip(X[num_p:], Y[num_p:]):
        #     f.write('%f,%f,'%(x, y))
def cut_bg(bp, img, raw_coords):
    cut_img = img [raw_coords [0] [1]:raw_coords [1] [1], raw_coords [0] [0]:raw_coords [1] [0]]
    cv2.imwrite (bp, cut_img)

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

def show(fp, lp):
    img = cv2.imread(fp)
    
    lps = read_label(fp, lp)
    for i in range(0,8,2):
        cv2.circle(img, (lps[i], lps [i+1]), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i/2) , (lps[i], lps [i+1]), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
    
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    img = cv2.resize(img, (800, 800))
    cv2.imshow('img', img)
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     exit(-1)

def main():
    f = open("cursor.txt", "rt")
    cur_index = int(f.readline())
    f.close() 
    j = 0
    print (cur_index)
    

    path = argv[1]
    save_path = argv[2]
    list_name = os.listdir(path)
    l = len (list_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range (cur_index, l):
        name = list_name [i]
        cur_index = i
        f_cursor = open("cursor.txt", "wt")
        f_cursor.write(str(cur_index))
        f_cursor.close()
        
        if 'txt' in name:
            continue
        fp = os.path.join(path, name)
        print(i ,'/',l, name)
        lp = os.path.join(save_path, name.split('.')[0] + '.txt')
        image = cv2.imread(fp)
        h, w, _ = image.shape
        if os.path.exists(lp):
            lps = read_label(fp, lp)
            if len (lps) == 8:
                for j in range(0,8,2):
                    cv2.circle(image, (lps[j], lps [j+1]), 5, (0, 0, 255), -1)
                    cv2.putText(image, str(int (j/2)) , (lps[j], lps [j+1]), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        # else:
        #     continue
        
       
        img = cv2.resize(image, (800, 800))
        cv2.imshow('img', img) 
        raw_coords, coords = [], []
        cv2.setMouseCallback('img', click, {'i':1, 'coords':coords, 'img':img, 'raw_coords':raw_coords})
        key = cv2.waitKey(0)
        if key == ord(' '):
            continue
        if key == ord('q'):
            exit(-1)
        if key == ord ('f'):
            write_label(lp, coords)
        
       
        # bp = os.path.join (save_path ,name.split('.')[0]+ '.jpg')
        # cut_bg(bp, img, raw_coords)
        

        

if __name__ == '__main__':
    main()





    #python3 label_tool.py '../data/labeled/Passport/add_padding_original' '../data/labeled/Passport/add_padding_original'