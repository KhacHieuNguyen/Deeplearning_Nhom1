import cv2
import numpy as np
import glob

src = glob.glob("data/01/*.txt")
dst = "data/Segmentation_annotation/"


for txt_file in src:
    img_file = txt_file[:-4] + ".jpg"
    print(img_file)
    img = cv2.imread(img_file)
    img = cv2.resize(img , (800,800))
    annotation = open(txt_file).readline()
    annotation = annotation.split(",")[:8]
    annotation = [float(i) for i in annotation]
    annotation = np.array(annotation).reshape(4,2)
    h,w = img.shape[:2]    
    coords = (annotation * np.array([h,w])).astype(np.uint16)
    src_coords = np.array([
        [0, 0], [w-1, 0], [w-1 , h-1], [0, h-1]], dtype="float32")
    trans, _ = cv2.findHomography(src_coords, coords)
    dst_map = cv2.warpPerspective(np.ones((800,800)), trans, (h,w)) 
    mask = np.array(dst_map > 0, dtype=np.uint8) * 255
    # print(mask)
    # mask = cv2.resize(mask , (img.shape[1],img.shape[0])) 
    
    # img= img.transpose((2,0,1))
    # mask = mask  * img
    # mask = mask.transpose((1,2,0))
   
    
    cv2.imwrite(dst + img_file.split("/")[-1] , mask)