from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.preprocessing.image import  load_img,img_to_array
from Verify import  trainData , imageFile
import numpy as np


# Data preprocessing
labels = trainData.iloc[:,1:].values


data = []
output = []
for ind in range(len(imageFile)):
    image = imageFile[ind]
    imageArray = cv2.imread(image)
    h,w,d = imageArray.shape
    # Preprocessing
    loadImage = load_img(image,target_size=(224,224))
    loadImageArray = img_to_array(loadImage)
    normLoadImageArray = loadImageArray/255.0 # normalization
    # Normalize the labels
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    labelNorm = (nxmin,nxmax,nymin,nymax) # normalized output
    #  Append
    data.append(normLoadImageArray)
    output.append(labelNorm)

x = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)

xTrain,xTest,yTrain,yTest = train_test_split(x,y,train_size=0.8,random_state=0)