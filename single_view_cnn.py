import os
import argparse
import numpy as np
import utility
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
EPOCHS = 10
BATCH_SIZE = 4

labels_dict = {'baththub': utility.int_to_1hot(0, 10),
              'bed': utility.int_to_1hot(1, 10),
              'chair': utility.int_to_1hot(2, 10),
              'desk': utility.int_to_1hot(3, 10),
              'dresser': utility.int_to_1hot(4, 10),
              'monitor': utility.int_to_1hot(5, 10),
              'night_stand': utility.int_to_1hot(6, 10),
              'sofa': utility.int_to_1hot(7, 10),
              'table': utility.int_to_1hot(8, 10),
              'toilet': utility.int_to_1hot(9, 10)}


def data_loader():
    for i in range(NUM_OBJECTS):
        file_path = os.path.join(DATA_PATH, FILES[i])
        x = keras.preprocessing.image.load_img(file_path,
                                               color_mode='rgb',
                                               target_size=(240, 320),
                                               interpolation='nearest')
        x = keras.preprocessing.image.img_to_array(x)
        label_class = labels_dict[FILES[i].split("_")[0]]
        label_view = utility.int_to_1hot(int(FILES[i].split("_")[-1].split(".")[0]), 60)
        yield x, (label_class, label_view)


def dataset_generator():
    dataset = tf.data.Dataset.from_generator(data_loader,
                                             output_types=(tf.float32, (tf.int16, tf.int16)),
                                             output_shapes=(tf.TensorShape([240, 320, 3]),
                                                            (tf.TensorShape([10]), tf.TensorShape([60]))))
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def generate_cnn():
    inputs = keras.Input(shape=(240, 320, 3))
    vgg = keras.applications.VGG16(include_top=False,
                                         weights='imagenet',
                                         input_tensor=inputs,
                                         input_shape=(240, 320, 3))
    preprocessed = keras.applications.vgg16.preprocess_input(inputs)
    x = vgg(preprocessed)
    x = layers.Flatten()(x)
    out_class = layers.Dense(10, activation='softmax', name="class")(x)
    out_view = layers.Dense(60, activation='softmax', name="view")(x)
    model = keras.Model(inputs=inputs, outputs=[out_class, out_view])
    model.summary()
    losses = {"class": "categorical_crossentropy",
              "view": "categorical_crossentropy"}
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=losses, metrics=METRICS[4:])
    # keras.utils.plot_model(model, "net_structure.png", show_shapes=True, expand_nested=True)
    return model


def main():
    # for idx, (x, (y1, y2)) in enumerate(dataset):
    #     print(idx, np.shape(x), y1, y2)

    model = generate_cnn()
    num_batches = int(NUM_OBJECTS / BATCH_SIZE)
    data_gen = dataset_generator()
    model.fit_generator(data_gen, steps_per_epoch=num_batches, epochs=EPOCHS)


if __name__ == '__main__':
    main()
