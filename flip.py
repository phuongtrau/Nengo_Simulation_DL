import cv2

image = cv2.imread("/home/hoangphuong/Desktop/cccd_recap/1635760153607.jpg")

image = cv2.flip(image,0)

cv2.imwrite("1635760153607.jpg",image)