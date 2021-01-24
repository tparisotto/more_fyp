import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import pandas as pd
import numpy as np
import utility
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# TODO: Split with validation/train set

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
CALLBACKS = [
    # tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='models/model_{epoch:02d}_{class_accuracy:.2f}_{view_accuracy:.2f}.h5',
                                       monitor='loss',
                                       mode='min',
                                       save_best_only=True,
                                       save_freq='epoch'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
EPOCHS = 3
BATCH_SIZE = 32

labels_dict = utility.get_label_dict()


def data_loader():
    for i in range(NUM_OBJECTS):
        file_path = os.path.join(DATA_PATH, FILES[i])
        x = keras.preprocessing.image.load_img(file_path,
                                               color_mode='rgb',
                                               target_size=(240, 320),
                                               interpolation='nearest')
        x = keras.preprocessing.image.img_to_array(x)
        label_class = FILES[i].split("_")[0]
        if label_class == 'night':
            label_class = 'night_stand'  # Quick fix for label parsing
        label_class = utility.int_to_1hot(labels_dict[label_class], 10)
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
    model = generate_cnn()
    num_batches = int(NUM_OBJECTS / BATCH_SIZE)
    data_gen = dataset_generator()
    history = model.fit(data_gen, steps_per_epoch=num_batches, epochs=EPOCHS, callbacks=CALLBACKS)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(f"class-view_training_history.csv"))


if __name__ == '__main__':
    main()
