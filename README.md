# Persian Vehicle License Plate Recognition

A Flask web application which written for  Vehicle License Plate Recognition wth **inceptionresnetv2** in Iran.

(As **mid-level** programmer you can easily change this behavior to English, French etc.)


### Final Appearance
![test.PNG](https://github.com/Ali-Fartout/Persian-Vehicle-License-Plate-Recognition/blob/master/test.PNG)


![Flowchart.PNG](https://github.com/Ali-Fartout/Persian-Vehicle-License-Plate-Recognition/blob/master/Flowchart.PNG)




## Packages and Libraries
### The libraries that I've used :

| Libraries | Links |
| ------ | ------ |
| TensorFlow| https://www.tensorflow.org |
| Numpy | https://numpy.org |
| Pandas | https://pandas.pydata.org |
| Matplotlib | https://matplotlib.org |
| CV2| https://opencv.org |
| Flask| https://flask.palletsprojects.com |
| Pytesseract | https://pypi.org/project/pytesseract |
| Scikit-Learn | https://scikit-learn.org |

And I've use this [repo](https://github.com/tzutalin/labelImg) for labeling my images and of course you can use as well.

## How to use?

After clone the repo.  first you **Have To**  run [Model.py](https://github.com/Ali-Fartout/Persian-Vehicle-License-Plate-Recognition/blob/master/Model.py) file to create model (you can also change hyperparameters). And then from Web App file run [app.py](https://github.com/Ali-Fartout/Persian-Vehicle-License-Plate-Recognition/blob/master/Web%20APP/app.py). Enjoy !

**Notice** :   As I said before, this application is for **Persian** . For changing language you have to add your data-language to your tesseract-ocr file. Then find some plates and label it with repo that I've said above. Drop the images to Images file. After that return xml file and with [XMLToCSV.py](https://github.com/Ali-Fartout/Persian-Vehicle-License-Plate-Recognition/blob/master/XMLToCSV.py) convert it to csv. Finally run Model and app file in order.

