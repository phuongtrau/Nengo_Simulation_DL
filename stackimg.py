import cv2
import numpy as np

# ls_non = ["13_non.jpg","7_non.jpg","5_non.jpg","12_non.jpg","15_non.jpg"]

# # ls_non = ["7_non.jpg","5_non.jpg","12_non.jpg"."15_non.jpg"]
# img_non = []
# img_pre = []
# for im in ls_non:
#     img_non.append(cv2.imread(im))
#     img_pre.append(cv2.imread(im.replace("_non","")))


# image_non = np.hstack(img_non)
# # print(image_non.shape)
# image_pre = np.hstack(img_pre)
# cv2.imwrite("img_non.jpg",image_non)
# img_out_gray = cv2.imread("img_non.jpg",0)
# cv2.imwrite("img_non_gray.jpg",img_out_gray)
# cv2.imwrite("img_pre.jpg",image_pre)
# img_out_gray = cv2.imread("img_pre.jpg",0)
# cv2.imwrite("img_pre_gray.jpg",img_out_gray)

# img_out = np.concatenate((image_non,image_pre),axis=0)
# print(img_out.shape)
# cv2.imwrite("im_out.jpg",img_out)
# img_out_gray = cv2.imread("im_out.jpg",0)
# cv2.imwrite("im_out_gray.jpg",img_out_gray)
# print("done")

img_model = cv2.imread("/home/hoangphuong/Desktop/photo_2021-10-28_10-02-17.jpg",0)
cv2.imwrite("model.jpg",img_model)