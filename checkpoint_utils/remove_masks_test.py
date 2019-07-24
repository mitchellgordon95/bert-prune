import tensorflow as tf
import numpy as np
import re
import sys
import os
from shutil import copyfile

if len(sys.argv) != 2:
    print("Usage: remove_masks_test.py [pretrain-dir]")
    print("Runs some sanity checks on the [pretrain-dir]_no_mask")
    exit()
else:
    CHECKPOINT_DIR = sys.argv[1].rstrip('/')
    OUTPUT_DIR = CHECKPOINT_DIR + "_no_mask"

with tf.Session() as sess:

    # Load all the variables from the checkpoint
    old_tensors = {}
    for var_name, _ in tf.train.list_variables(CHECKPOINT_DIR):
        old_tensors[var_name] = tf.contrib.framework.load_variable(CHECKPOINT_DIR, var_name)

    # Load all the variables from the checkpoint
    new_tensors = {}
    for var_name, _ in tf.train.list_variables(OUTPUT_DIR):
        new_tensors[var_name] = tf.contrib.framework.load_variable(OUTPUT_DIR, var_name)

    assert set(new_tensors.keys()) == set(old_tensors.keys())

    for var_name, tensor in new_tensors.items():
        # All the masks should be ones
        if var_name.endswith('/mask'):
            assert np.array_equal(tensor, np.ones_like(tensor)), f'{var_name} {tensor}'

        # If we multiply the new weights by the old mask again, it shouldn't change the value
        if var_name.endswith('/weights'):
            mask_name = re.sub(r'weights', r'mask', var_name)
            assert np.array_equal(tensor, tensor * old_tensors[mask_name])

    embed = new_tensors['bert/embeddings/word_embeddings']
    mask_name ='bert/embeddings/embed_mask/mask'
    assert np.array_equal(embed, embed * old_tensors[mask_name])
