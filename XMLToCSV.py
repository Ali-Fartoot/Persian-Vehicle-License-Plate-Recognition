import pandas as pd
import xml.etree.ElementTree as xet
from glob import glob

# Return address xml files from jpeg files
XML_PATH = glob("./Images/*.xml")
labelsDict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])
for filename in XML_PATH:

    info = xet.parse(filename)
    root = info.getroot()
    memberObject = root.find('object')
    labelsInfo = memberObject.find('bndbox')
    # Append xmin xmax ymin ymax label from xml files to dict
    xmin = int(labelsInfo.find('xmin').text)
    xmax = int(labelsInfo.find('xmax').text)
    ymin = int(labelsInfo.find('ymin').text)
    ymax = int(labelsInfo.find('ymax').text)

    labelsDict['filepath'].append(filename)
    labelsDict['xmin'].append(xmin)
    labelsDict['xmax'].append(xmax)
    labelsDict['ymin'].append(ymin)
    labelsDict['ymax'].append(ymax)

# Make Dataset
trainData = pd.DataFrame(labelsDict)
trainData.to_csv('Labels.csv',index=False)