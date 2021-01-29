import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
import scipy.signal as sig
import utility
from tqdm import tqdm, trange
import trimesh
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
print(f"Tensorflow v{tf.__version__}\n")

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, help="Folder with view-dataset")
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('-p', '--points', type=int, default=2048, help="Sample points of the point cloud.")
parser.add_argument('-s', '--split', type=float, default=0.9)
parser.add_argument('--sample_rate', type=float, default=10, help="Filter factor for the view-dataset")
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--out', default="./")
parser.add_argument('--save_csv')
parser.add_argument('--save_npz')
parser.add_argument('--load_csv')
parser.add_argument('--load_npz')
parser.add_argument('--modelnet_path')
args = parser.parse_args()

TIMESTAMP = datetime.now().strftime('%d-%m-%H%M')
BASE_DIR = sys.path[0]
DATA_DIR = args.data
MN_DIR = args.modelnet_path
MODEL_DIR = os.path.join(args.out, f"PointNet-{TIMESTAMP}")
NUM_VIEWS = 60
SAMPLE_RATE = int(args.sample_rate)
SPLIT = args.split
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
    # tf.keras.callbacks.EarlyStopping(patience=3),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'pointnet_epoch-{epoch:02d}_loss-{val_loss:.3f}.h5'),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_freq='epoch'),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(MODEL_DIR, 'logs/')),
    # tf.keras.callbacks.LearningRateScheduler(scheduler)
]


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        return {'num_features': int(self.num_features), 'l2reg': float(self.l2reg)}

def parse_data():
    files = os.listdir(DATA_DIR)
    print(f"[INFO] Parsing data from {DATA_DIR}...")
    files.sort()
    for filename in files:  # Removes file without .png extension
        if not filename.endswith('png'):
            files.remove(filename)
    # files = files[::SAMPLE_RATE]

    labels = []
    object_index = []
    phi = []
    theta = []
    view_code = []
    entropy = []
    if args.verbose:
        for filename in tqdm(files):
            value_string = filename.replace(".png", "")
            label = value_string.split("_")[0]
            if label == "night":
                label = "night_stand"
            labels.append(label)
            value_string = value_string.split("_")
            view_code.append(int(value_string[-1]))
            phi.append(int(value_string[-3]))
            theta.append(int(value_string[-5]))
            object_index.append(value_string[-7])
            img = plt.imread(os.path.join(DATA_DIR, filename))
            entropy.append(shannon_entropy(img))
    else:
        for filename in files:
            value_string = filename.replace(".png", "")
            label = value_string.split("_")[0]
            if label == "night":
                label = "night_stand"
            labels.append(label)
            value_string = value_string.split("_")
            view_code.append(int(value_string[-1]))
            phi.append(int(value_string[-3]))
            theta.append(int(value_string[-5]))
            object_index.append(value_string[-7])
            img = plt.imread(os.path.join(DATA_DIR, filename))[:, :, 0]  # every channel is the same
            entropy.append(shannon_entropy(img))
    df = pd.DataFrame({"label": labels,
                       "object_index": object_index,
                       "view_code": view_code,
                       "phi": phi,
                       "theta": theta,
                       "entropy": entropy})
    return df


def extract_entropy_peaks(data):
    maxima = []
    maxima_entropy = []
    for i in range(0, 5):
        sub = data[data['theta'] == 30 * (i + 1)]
        x = np.array(sub['phi'])
        y = np.array(sub['entropy'])
        sub = zip(x, y)
        subsort = sorted(sub)
        tuples = zip(*subsort)
        x, y = [np.array(el) for el in tuples]
        # Trick to have the first or last element of the signal considered as a peak candidate
        ext, _ = sig.find_peaks(np.concatenate(([y[-1]], y, [y[0]])))
        ext = ext - 1
        maxima.append(ext)
        maxima_entropy.append(y)
    maxima = np.array(maxima)
    maxima_entropy = np.array(maxima_entropy)
    views = np.zeros(12)
    views_entropy = np.zeros(12)
    for j in range(0, 5):
        for i in range(0, 12):
            if i in maxima[j]:
                if maxima_entropy[j, i] > views_entropy[i]:
                    views_entropy[i] = maxima_entropy[j, i]
                    views[i] = j + 1
    labels = []
    for i in range(0, 12):
        if views[i] > 0:
            labels.append((views[i] * 30.0, i * 30.0))
    return labels


def get_vectors_from_view_labels(data):
    ### Dictionaries ###
    label2idx = {}
    count = 0
    for theta in range(30, 151, 30):
        for phi in range(0, 331, 30):
            label2idx[theta, phi] = count
            count += 1

    idx2label = {}
    count = 0
    for theta in range(30, 151, 30):
        for phi in range(0, 331, 30):
            idx2label[count] = (theta, phi)
            count += 1
    #######################
    subset_labels = extract_entropy_peaks(data)
    subset_idx = []
    for lab in subset_labels:
        subset_idx.append(label2idx[lab])
    subset_idx.sort()
    return subset_idx


def load_dataset(data, num_points):
    num_objects = int(len(data) / NUM_VIEWS)
    x = []
    y = []
    if args.verbose:
        for i in trange(num_objects):
            if i % SAMPLE_RATE == 0:
                entry = data.iloc[NUM_VIEWS * i]
                file_path = os.path.join(MN_DIR, entry.label, "train", f"{entry.label}_{entry.object_index:04}.off")
                mesh = trimesh.load(file_path)
                bound = float(max(np.abs(np.max(mesh.vertices)), np.abs(np.min(mesh.vertices))))
                mesh.apply_scale(1 / bound)  # Normalization within [-1, 1]
                x_point_cloud = mesh.sample(num_points)
                x.append(x_point_cloud)
        for i in trange(num_objects):
            if i % SAMPLE_RATE == 0:
                data_subset = data.iloc[
                              NUM_VIEWS * i:NUM_VIEWS * (i + 1)]  # each object is represented by $NUM_VIEWS entries
                labels = get_vectors_from_view_labels(data_subset)
                label_vectors = utility.view_vector(labels, NUM_VIEWS)
                y.append(label_vectors)
    else:
        for i in range(num_objects):
            if i % SAMPLE_RATE == 0:
                entry = data.iloc[NUM_VIEWS * i]
                file_path = os.path.join(MN_DIR, entry.label, "train", f"{entry.label}_{entry.object_index:04}.off")
                mesh = trimesh.load(file_path)
                bound = float(max(np.abs(np.max(mesh.vertices)), np.abs(np.min(mesh.vertices))))
                mesh.apply_scale(1 / bound)  # Normalization within [-1, 1]
                x_point_cloud = mesh.sample(num_points)
                x.append(x_point_cloud)
        for i in range(num_objects):
            if i % SAMPLE_RATE == 0:
                data_subset = data.iloc[
                              NUM_VIEWS * i:NUM_VIEWS * (i + 1)]  # each object is represented by $NUM_VIEWS entries
                labels = get_vectors_from_view_labels(data_subset)
                label_vectors = utility.view_vector(labels, NUM_VIEWS)
                y.append(label_vectors)
    return x, y


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def tnet(inputs, num_features):
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


def generate_pointnet():
    inputs = keras.Input(shape=(args.points, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_VIEWS, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        metrics=METRICS,
    )
    model.summary()

    return model


def main():
    if args.load_csv is None:
        data = parse_data()
    else:
        data = pd.read_csv(args.load_csv)
        print("[INFO] CSV correctly loaded.")
    if args.save_csv is not None:
        data.to_csv(args.save_csv, index=False)
        print(f"[INFO] CSV saved in {args.save_csv}")

    if args.load_npz is None:
        x, y = load_dataset(data, args.points)
    else:
        x = np.load(args.load_npz)["x"]
        y = np.load(args.load_npz)["y"]
        print("[INFO] Dataset correctly loaded.")
    if args.save_npz is not None:
        np.savez_compressed(args.save_npz, x=x, y=y)
        print(f"[INFO] Numpy dataset saved in {args.save_npz}")

    model = generate_pointnet()
    history = model.fit(x, y,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        validation_split=args.split,
                        callbacks=CALLBACKS)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(MODEL_DIR, f"pointnet_history-{TIMESTAMP}.csv"))
    print(f"[INFO] Model and logs saved in {MODEL_DIR}")


if __name__ == '__main__':
    main()
