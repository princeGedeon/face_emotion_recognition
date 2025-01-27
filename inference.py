import cv2
import numpy as np
import tensorflow as tf

def init_model():
    return tf.keras.models.load_model("model.h5")

def inference(img,model,shape=(48,48)):
    img_resize = cv2.resize(img, shape)

    img_resize = np.expand_dims(img_resize, axis=0)

    pred = model.predict(img_resize)
    rec = pred.argmax(axis=1)
    return rec