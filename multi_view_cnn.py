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
    true_labels = []
    true_views = []
    for file in args.views:
        images.append(cv2.imread(file))
        true_labels.append(os.path.split(file.split("_")[-8])[-1])
        true_views.append(file.split("_")[-1].split(".")[0])
    model = keras.models.load_model(args.model)
    images = np.array(images)
    results = model.predict(images)
    labels = results[0]
    views = results[1]
    dic = utility.get_label_dict(inverse=True)
    for i in range(len(views)):
        print(f"Predicted: {dic[np.argmax(labels[i])]}, {np.argmax(views[i])} - True: {true_labels[i]}, {true_views[i]}")


if __name__ == "__main__":
    main()
