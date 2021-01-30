import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import utility

parser = argparse.ArgumentParser()
parser.add_argument('--views', nargs='+')
parser.add_argument('--model')
args = parser.parse_args()


# TODO: load x models where x is the number of views received from the entropy model
#     ensemble the results of the x networks, class=voting, pose=infer_from_views


def main():
    print("[INFO] Prediction results:")
    images = []
    for file in args.views:
        images.append(cv2.imread(file))
    model = keras.models.load_model(args.model)
    images = np.array(images)
    results = model.predict(images)
    labels = results[0]
    views = results[1]
    dic = utility.get_label_dict(inverse=True)
    for i in range(len(views)):
        print(dic[np.argmax(labels[i])], np.argmax(views[i]))


if __name__ == "__main__":
    main()
