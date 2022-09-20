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

def get_batch_of_random_lc_data(seq_len, batch_size):
    x_batch = np.zeros((batch_size, seq_len, 2))
    y_batch = np.zeros((batch_size, seq_len, 2))

    for i in range(batch_size):
        x0 = np.random.rand(2)
        lc = limitcycles.SimpleLimitCycle(0, 0)
        seq = lc.get_nsteps(x0, 0.1, seq_len+1)
        xseq, yseq = seq[:-1], seq[1:]

        x_batch[i] = xseq
        y_batch[i] = yseq

    return x_batch, y_batch

if __name__ == '__main__':
    print(get_batch_of_random_lc_data(5, 1))
