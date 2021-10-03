from tensorflow.keras.applications import MobileNetV2, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from DataPreprocessing import xTrain,xTest,yTrain,yTest

# Using InceptionResNetV2 model from tensorflow
inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False,
                                     input_tensor=Input(shape=(224,224,3)))

# Freeze layers
inception_resnet.trainable=False

# Complete model
headModel = inception_resnet.output
headModel = Flatten()(headModel)
headModel = Dense(500,activation="relu")(headModel)
headModel = Dense(250,activation="relu")(headModel)
headModel = Dense(4,activation='sigmoid')(headModel)

model = Model(inputs=inception_resnet.input,outputs=headModel)


# Compile model
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
model.summary()

# Train model
tfb = TensorBoard('object_detection')
history = model.fit(x=xTrain,y=yTrain,batch_size=10,epochs=200,
                    validation_data=(xTest,yTest),callbacks=[tfb])

model.save('./Web App/static/models/model.h5')