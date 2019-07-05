import tensorflow as tf
import modeling
import re
import sys
import os
from tensorflow.python.training import checkpoint_management
from shutil import copyfile

if len(sys.argv) < 2:
    print("Usage: fix_embed_mask_name.py [chkpt-dir]")
    print("Fix the latest checkpoint in the directory by renaming the embedding mask variable.")
    print("Note: overwrites original checkpoint!")
    exit()
else:
    CHECKPOINT_DIR = sys.argv[1].rstrip('/')
    CHECKPOINT_FILE = checkpoint_management.latest_checkpoint(CHECKPOINT_DIR)

with tf.Session() as sess:

    # Load all the variables from the checkpoint, renaming as we go
    for var_name, _ in tf.train.list_variables(CHECKPOINT_FILE):
        var_tensor = tf.contrib.framework.load_variable(CHECKPOINT_FILE, var_name)

        new_name = re.sub(r'bert/embeddings//mask', r'bert/embeddings/embed_mask/mask', var_name)
        new_name = re.sub(r'bert/embeddings//threshold', r'bert/embeddings/embed_mask/threshold', new_name)

        if new_name != var_name:
            print(f"Renaming {var_name} to {new_name}")

        var = tf.Variable(var_tensor, name=new_name)

    # Save these new variables
    print(f'Writing to {CHECKPOINT_FILE}')
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, CHECKPOINT_FILE)
