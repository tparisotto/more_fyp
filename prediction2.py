import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser()
parser.add_argument("picture")
args = parser.parse_args()

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

model = keras.models.load_model("model_checkpoint.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=METRICS[4:])

file_path = args.picture
x = keras.preprocessing.image.load_img(file_path,
                                               color_mode='rgb',
                                               target_size=(240, 320),
                                               interpolation='nearest')
x = keras.preprocessing.image.img_to_array(x)
x = x.reshape((1,240,320, 3))


print("[INFO] Computing prediction...")
y1pred, y2pred = model.predict(x)
y1out, y2out = np.argmax(y1pred), np.argmax(y2pred)
print(y1out)
print(y2out)
# print(yout)
# print(f"Accuracy: {1 - np.linalg.norm(y - yout) / 60}")
