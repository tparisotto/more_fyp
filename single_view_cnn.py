import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser()
parser.add_argument("data", help="Directory where the single views are stored.")
args = parser.parse_args()

DATA_PATH = args.data
FILES = os.listdir(DATA_PATH)
for filename in FILES:  # Removes file without .png extension
    if not filename.endswith('png'):
        FILES.remove(filename)
NUM_OBJECTS = len(FILES)

labels_int = {'baththub': 0,
              'bed': 1,
              'chair': 2,
              'desk': 3,
              'dresser': 4,
              'monitor': 5,
              'night_stand': 6,
              'sofa': 7,
              'table': 8,
              'toilet': 9}


def data_loader():
    for i in range(NUM_OBJECTS):
        file_path = os.path.join(DATA_PATH, FILES[i])
        x = keras.preprocessing.image.load_img(file_path,
                                               color_mode='grayscale',
                                               target_size=(240, 320),
                                               interpolation='nearest')
        x = keras.preprocessing.image.img_to_array(x)
        label_class = labels_int[FILES[i].split("_")[0]]
        label_view = int(FILES[i].split("_")[-1].split(".")[0])
        yield x, (label_class, label_view)


def main():
    dataset = tf.data.Dataset.from_generator(data_loader,
                                             output_types=(tf.float32, (tf.int16, tf.int16)),
                                             output_shapes=(tf.TensorShape([240, 320]),
                                                            (tf.TensorShape([1]), tf.TensorShape([1]))))

    iterator = dataset.make_one_shot_iterator()
    x, (y1, y2) = iterator.get_next()
    print(np.shape(x), np.shape(y1), np.shape(y2))

if __name__ == '__main__':
    main()
