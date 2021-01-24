import os
import argparse
import tensorflow as tf
import tensorflow.keras

# TODO: load x models where x is the number of views received from the entropy model
#     ensemble the results of the x networks, class=voting, pose=infer_from_views
