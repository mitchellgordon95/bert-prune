import tensorflow as tf
import numpy as np
import re
import sys
import os
from shutil import copyfile

if len(sys.argv) != 2:
    print("Usage: memory_breakdown.py [pretrain-dir]")
    exit()

with tf.Session() as sess:

    # Load all the variables from the checkpoint
    total = 0
    embeddings = 0
    attention = 0
    FC = 0
    other = 0
    masks = 0
    cls = 0
    for var_name, _ in tf.train.list_variables(sys.argv[1]):
        tensor = tf.contrib.framework.load_variable(sys.argv[1], var_name)

        total += tensor.size

        if var_name.endswith('/mask'):
            masks += tensor.size
        elif var_name.startswith('cls'):
            cls += tensor.size
        elif 'embeddings/word_embeddings' in var_name:
            embeddings += tensor.size
        elif '/attention/' in var_name:
            attention += tensor.size
        elif '/intermediate' in var_name or '/output/' in var_name:
            FC += tensor.size
        else:
            other += tensor.size

    total -= masks
    print(f"""
    Embeds: {embeddings} ({int(embeddings/total * 100)}%)
    Attention: {attention} ({int(attention/total * 100)}%)
    FC: {FC} ({int(FC/total * 100)}%)
    cls: {cls} ({int(cls/total * 100)}%)
    other: {other} ({int(other/total * 100)}%)
    Total: {total}

    (masks: {masks})
    """)

