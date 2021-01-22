
import os
import argparse
from tqdm import trange
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("out")
args = parser.parse_args()


DATA_PATH = args.data
OUT_PATH = args.out

files = os.listdir(DATA_PATH)
for filename in files:  # Removes file without .png extension
    if not filename.endswith('png'):
        files.remove(filename)
num_objects = len(files)

x = []
y = []
print(f"[INFO] Processing {num_objects} images into numpy arrays...")
for i in trange(5):
    filepath = os.path.join(DATA_PATH, files[i])
    image = Image.open(filepath)
    x.append()
    y.append()

# np.save('x_views_data.npy', x)
# np.save('y_views_data.npy', y)
# print("[INFO] Features (x_views_data.npy) and Labels (y_views_data.npy) successfully stored in current directory.")