import tensorflow as tf
import numpy as np
import re
import sys
import os
from shutil import copyfile

if len(sys.argv) != 2:
    print("Usage: remove_masks.py [pretrain-dir]")
    print("Creates an output directory = [pretrain-dir]_no_mask")
    exit()
else:
    CHECKPOINT_DIR = sys.argv[1].rstrip('/')
    OUTPUT_DIR = CHECKPOINT_DIR + "_no_mask"

os.mkdir(OUTPUT_DIR)

with tf.Session() as sess:

    # Load all the variables from the checkpoint
    tensors = {}
    for var_name, _ in tf.train.list_variables(CHECKPOINT_DIR):
        tensors[var_name] = tf.contrib.framework.load_variable(CHECKPOINT_DIR, var_name)

    # For each mask variable
    for mask_name in filter(lambda x: x.endswith('/mask'), tensors.keys()):
        if mask_name == 'bert/embeddings/embed_mask/mask':
            weight_name = 'bert/embeddings/word_embeddings'
        else:
            weight_name = re.sub(r'mask', r'weights', mask_name)

        weight = tensors[weight_name]
        mask = tensors[mask_name]

        tensors[weight_name] = mask * weight
        # Use np.ones_like instead of np.ones here to avoid changing the dtype of the
        # array. By default, np.ones dtype is float64. This is twice the size of
        # the original float32, which overflows the protocol buffer.
        tensors[mask_name] = np.ones_like(mask)

    for var_name, var_tensor_np in tensors.items():
        var = tf.Variable(var_tensor_np, name=var_name)

    # Save these new variables
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, os.path.join(OUTPUT_DIR, 'no_mask.ckpt'))
