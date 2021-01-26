import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import pandas as pd
import numpy as np
import utility
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("train_data")
parser.add_argument("test_data")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("-a", "--architecture", default="vgg",
                    choices=['efficientnet', 'vgg', 'mobilenet', 'mobilenetv2', 'light'])
parser.add_argument("-o", "--out", default="./")
args = parser.parse_args()

TIMESTAMP = utility.get_datastamp()
MODEL_DIR = os.path.join(args.out, f"{args.architecture}-{TIMESTAMP}")

TRAIN_DATA_PATH = args.train_data
TRAIN_FILES = os.listdir(TRAIN_DATA_PATH)
for filename in TRAIN_FILES:  # Removes file without .png extension
    if not filename.endswith('png'):
        TRAIN_FILES.remove(filename)
np.random.shuffle(TRAIN_FILES)
NUM_OBJECTS_TRAIN = len(TRAIN_FILES)
TRAIN_FILTER = 1000

TEST_DATA_PATH = args.test_data
TEST_FILES = os.listdir(TEST_DATA_PATH)
for filename in TEST_FILES:
    if not filename.endswith('png'):
        TEST_FILES.remove(filename)
np.random.shuffle(TEST_FILES)
NUM_OBJECTS_TEST = len(TEST_FILES)
TEST_FILTER = 100

os.makedirs(MODEL_DIR)

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


# def scheduler(epoch, lr):
#     if epoch < 1:
#         return lr
#     else:
#         return lr * tf.math.exp(-1.099)


CALLBACKS = [
    # tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'model_epoch-{epoch:02d}_ca-{class_accuracy:.2f}_'
                                         'va-{view_accuracy:.2f}.h5'),
        monitor='loss',
        mode='min',
        save_best_only=True,
        save_freq='epoch'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    # tf.keras.callbacks.LearningRateScheduler(scheduler)
]
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size


def data_loader_train():
    labels_dict = utility.get_label_dict()
    for i in range(NUM_OBJECTS_TRAIN):
        if i % TRAIN_FILTER == 0:
            file_path = os.path.join(TRAIN_DATA_PATH, TRAIN_FILES[i])
            x = keras.preprocessing.image.load_img(file_path,
                                                   color_mode='rgb',
                                                   target_size=(240, 320),
                                                   interpolation='nearest')
            x = keras.preprocessing.image.img_to_array(x)
            label_class = TRAIN_FILES[i].split("_")[0]
            if label_class == 'night':
                label_class = 'night_stand'  # Quick fix for label parsing
            label_class = utility.int_to_1hot(labels_dict[label_class], 10)
            label_view = utility.int_to_1hot(int(TRAIN_FILES[i].split("_")[-1].split(".")[0]), 60)
            yield x, (label_class, label_view)


def data_loader_test():
    labels_dict = utility.get_label_dict()
    for i in range(NUM_OBJECTS_TEST):
        if i % TEST_FILTER == 0:
            file_path = os.path.join(TEST_DATA_PATH, TEST_FILES[i])
            x = keras.preprocessing.image.load_img(file_path,
                                                   color_mode='rgb',
                                                   target_size=(240, 320),
                                                   interpolation='nearest')
            x = keras.preprocessing.image.img_to_array(x)
            label_class = TEST_FILES[i].split("_")[0]
            if label_class == 'night':
                label_class = 'night_stand'  # Quick fix for label parsing
            label_class = utility.int_to_1hot(labels_dict[label_class], 10)
            label_view = utility.int_to_1hot(int(TEST_FILES[i].split("_")[-1].split(".")[0]), 60)
            yield x, (label_class, label_view)


def dataset_generator_train():
    dataset = tf.data.Dataset.from_generator(data_loader_train,
                                             output_types=(tf.float32, (tf.int16, tf.int16)),
                                             output_shapes=(tf.TensorShape([240, 320, 3]),
                                                            (tf.TensorShape([10]), tf.TensorShape([60]))))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(EPOCHS)
    return dataset


def dataset_generator_test():
    dataset = tf.data.Dataset.from_generator(data_loader_test,
                                             output_types=(tf.float32, (tf.int16, tf.int16)),
                                             output_shapes=(tf.TensorShape([240, 320, 3]),
                                                            (tf.TensorShape([10]), tf.TensorShape([60]))))
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def generate_cnn(app="efficientnet"):
    inputs = keras.Input(shape=(240, 320, 3))

    if app == "vgg":
        net = keras.applications.VGG16(include_top=False,
                                       weights='imagenet',
                                       input_tensor=inputs)
        preprocessed = keras.applications.vgg16.preprocess_input(inputs)
        x = net(preprocessed)

    elif app == "efficientnet":
        net = keras.applications.EfficientNetB0(include_top=False,
                                                weights='imagenet')
        preprocessed = keras.applications.efficientnet.preprocess_input(inputs)
        x = net(preprocessed)

    elif app == "mobilenet":
        net = keras.applications.MobileNet(include_top=False,
                                           weights='imagenet')
        preprocessed = keras.applications.mobilenet.preprocess_input(inputs)
        x = net(preprocessed)

    elif app == "mobilenetv2":
        net = keras.applications.MobileNetV2(include_top=False,
                                             weights='imagenet')
        preprocessed = keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = net(preprocessed)

    elif app == "light":
        x = keras.layers.Conv2D(32, 5, 3, activation='relu')(inputs)
        x = keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = keras.layers.Conv2D(32, 5, 3, activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(3, 3))(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    out_class = layers.Dense(10, activation='softmax', name="class")(x)
    out_view = layers.Dense(60, activation='softmax', name="view")(x)
    model = keras.Model(inputs=inputs, outputs=[out_class, out_view])
    model.summary()
    losses = {"class": 'categorical_crossentropy',
              "view": 'categorical_crossentropy'}
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=losses, metrics=METRICS[4:])
    # keras.utils.plot_model(model, "net_structure.png", show_shapes=True, expand_nested=True)
    return model


def main():
    model = generate_cnn(app=args.architecture)
    num_batches = int(NUM_OBJECTS_TRAIN / BATCH_SIZE)
    train_data_gen = dataset_generator_train()
    test_data = list(dataset_generator_test().as_numpy_iterator())
    history = model.fit(train_data_gen,
                        shuffle=True,
                        steps_per_epoch=num_batches,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=CALLBACKS,
                        validation_data=test_data)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(f"class-view_training_history.csv"))


if __name__ == '__main__':
    main()
