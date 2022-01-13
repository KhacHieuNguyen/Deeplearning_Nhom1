import cv2
import numpy as np 
import glob
def check_annotation(image_file):
    image = cv2.imread(image_file)
    annotation = open(image_file[:-4] + ".txt").readline()
    annotation = annotation.split(",")[:8]
    annotation = [float(i) for i in annotation]
    annotation = np.array(annotation).reshape(4,2)
    img_shape = image.shape
    print(img_shape)
    i = 0
    for point in annotation:
        y = int(point[0] * image.shape[1])
        x = int(point[1] * image.shape[0])
        print(y,x)
        image = cv2.circle(image , (y,x) , radius = 5,color = (255,255,0) ,thickness=1)
        cv2.putText(image, str(i) , (y, x), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        i+=1
    return image

def test_image(image, annotation):
    annotation = np.array(annotation).reshape(4,2)
    img_shape = image.shape
    i=0
    for point in annotation:
        y = int(point[0] * image.shape[1])
        x = int(point[1] * image.shape[0])
        image = cv2.circle(image , (y,x) , radius = 5,color = (255,0,0) ,thickness=5)
        cv2.putText(image, str(i) , (y, x), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        i+=1
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    return image
def get_RoI(image_file, annotation = None):
    if type(image_file) == type("123.jpg"):
        image = cv2.imread(image_file)
    else:
        image = image_file
    if annotation is None:
        annotation = open(image_file[:-4] + ".txt").readline()
        annotation = annotation.split(",")[:8]
        annotation = [float(i) for i in annotation]
        annotation = np.array(annotation).reshape(4,2)
    img_shape = image.shape
    order_point = []
    for point in annotation:
        y = int(point[0] * image.shape[1])
        x = int(point[1] * image.shape[0])
        order_point.append((y,x))
    (pt_A, pt_B,pt_C,pt_D) = tuple(order_point)
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
    
    
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out = cv2.warpPerspective(image,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
    return out

if __name__ == "__main__":

    folder = glob.glob("annotations/test/*.jpg") + glob.glob("annotations/train/*.jpg")
    print(len(folder))
    for filename in folder:
        print(filename)
        image = cv2.imread(filename)
        image = cv2.resize(image , (1600,1600))
        cv2.imwrite(filename,image)