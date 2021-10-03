import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt
import os

# Load model
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, os.path.join('static', 'roi'))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 2))])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

model = tf.keras.models.load_model('./static/models/model.h5')


def ObjectDetection(PATH, FILENAME):
    # read image
    image = load_img(PATH)
    # PIL object
    image = np.array(image, dtype=np.uint8)
    # 8 bit array (0,255)
    image1 = load_img(PATH, target_size=(224, 224))
    # data preprocessing
    normImageArray = img_to_array(image1) / 255.0
    # convert into array and get the normalized output
    h, w, d = image.shape
    test_arr = normImageArray.reshape(1, 224, 224, 3)
    # make predictions
    predictions = model.predict(test_arr)
    # denormalize the values
    denorm = np.array([w, w, h, h])
    predictions = predictions * denorm
    predictions = predictions.astype(np.int32)
    # draw bounding on top the image
    xmin, xmax, ymin, ymax = predictions[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./Web APP/static/Predict/{}'.format(FILENAME), image_bgr)
    return predictions

# Get picture from object detection and convert it to text
def OCR(path, filename):
    image = np.array(load_img(path))
    predictions = ObjectDetection(path, filename)
    xmin, xmax, ymin, ymax = predictions[0]
    roi = image[ymin:ymax, xmin:xmax]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(UPLOAD_PATH, filename), roi_bgr)
    # For Persian i should config. following code to my cv2 config.
    # Maybe you HAVE TO change the direction.
    tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'
    text = pt.image_to_string(roi, lang='fas', config=tessdata_dir_config)
    print(text)
    return text
