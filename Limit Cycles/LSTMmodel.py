from xml.etree.ElementInclude import include
import numpy as np
import matplotlib.pyplot as plt
import limitcycles
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["PYTHONHASHSEED"] = "0"
import tensorflow as tf
from tqdm import tqdm

np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

# ----------------------- CREATING THE MODEL -----------------------

def get_model(num_units=2, include_dense=False, init_diag=1, batch_size=1):
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            num_units, 
            activation='tanh', 
            #input_shape=(seqlen, 2),
            kernel_initializer=tf.keras.initializers.Identity(init_diag),
            recurrent_initializer=tf.keras.initializers.Identity(init_diag),
            stateful=True,
            return_sequences=True,
            batch_input_shape=(batch_size, None, 2)
        )
    ])

    if include_dense:
        model.add(tf.keras.layers.Dense(2))
    
    return model

