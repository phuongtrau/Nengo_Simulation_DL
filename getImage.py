import cv2
from utils import *

exp_i_data = export_data_exp_i("experiment-i")
exp_i_data_non = export_data_exp_i("experiment-i",preprocess=False)
datasets = {"Base":exp_i_data}

data = Mat_Dataset(datasets,["Base"],["S1"])
img = []
for i in range (len(data.samples)):
    if data.labels[i]==0 and len(img)==0:
        img.append(data.samples[i])
    elif data.labels[i]==1 and len(img)<1:
        img.append(data.samples[i])
    elif data.labels[i]==2 and len(img)<2:
        img.append(data.samples[i])
    elif data.labels[i]==3 and len(img)<3:
        img.append(data.samples[i])
    elif data.labels[i]==4 and len(img)<4:
        img.append(data.samples[i])
    elif data.labels[i]==5 and len(img)<5:
        img.append(data.samples[i])
    elif data.labels[i]==6 and len(img)<6:
        img.append(data.samples[i])
    elif data.labels[i]==7 and len(img)<7:
        img.append(data.samples[i])
    elif data.labels[i]==8 and len(img)<8:
        img.append(data.samples[i])
    elif data.labels[i]==9 and len(img)<9:
        img.append(data.samples[i])
    elif data.labels[i]==10 and len(img)<10:
        img.append(data.samples[i])
    elif data.labels[i]==11 and len(img)<11:
        img.append(data.samples[i])
    elif data.labels[i]==12 and len(img)<12:
        img.append(data.samples[i])
    elif data.labels[i]==13 and len(img)<13:
        img.append(data.samples[i])
    elif data.labels[i]==14 and len(img)<14:
        img.append(data.samples[i])
    elif data.labels[i]==15 and len(img)<15:
        img.append(data.samples[i])
    elif data.labels[i]==16 and len(img)<16:
        img.append(data.samples[i])
        # break
print(len(img))
for i in range(len(img)):
    cv2.imwrite(str(i)+".jpg",cv2.applyColorMap(img[i], cv2.COLORMAP_PINK))