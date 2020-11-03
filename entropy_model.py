import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silence warnings

# TODO: The model should predict a 40-dimension vector
#  with the positions that corresponds to values over the 90% percentile of the entropy


print(tf.__version__)