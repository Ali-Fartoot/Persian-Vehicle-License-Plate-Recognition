import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import xml.etree.ElementTree as xet
from XMLToCSV import labelsDict

# Make Dataset
trainData = pd.DataFrame(labelsDict)
trainData.to_csv('Labels.csv',index=False)


# Data preprocessing
labels = trainData.iloc[:,1:].values


# Return address of jpeg files from each xml file

def GetFileName(filename):
    imageFileName = xet.parse(filename).getroot().find('filename').text
    return os.path.join('./images', imageFileName)


# Reuse Labels data
trainData = pd.read_csv('Labels.csv')

# Apply our function
imageFile = list(trainData['filepath'].apply(GetFileName))

# Verify image

if __name__ == "__main__":
    img = cv2.imread(imageFile[0])

    # Make rectangle for plates
    cv2.rectangle(img,(260,266),(584,330),(0,255,0),3)
    cv2.namedWindow('example',cv2.WINDOW_NORMAL)
    cv2.imshow('example',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()