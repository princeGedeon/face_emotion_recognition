### Exportation
"""import tensorflow as tf
import tf2onnx
import onnx
from tensorflow import keras
from tensorflow.keras.models import load_model

model = load_model('/gdrive/MyDrive/ml/model2.h5')

input_signature = [tf.TensorSpec([None,112, 112,3], tf.float32, name='x')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13, output_path="model.onnx")

output_names = [n.name for n in onnx_model.graph.output]
"""

### Inférence
import cv2
import numpy as np

import onnxruntime as rt
providers = ['CPUExecutionProvider']



def init_model():
    return rt.InferenceSession("model.onnx", providers=providers)

def inference(img,model,shape=(112,112),output_names=['dense_1']):
    img_resize = cv2.resize(img, shape)
    img_resize=img_resize / 255.0
    img_resize=img_resize.reshape(1,112,112,3)

    pred = model.run(output_names, {"x": img_resize.astype(np.float32)})
    rec = np.argmax(pred[0])
    return (rec,pred)