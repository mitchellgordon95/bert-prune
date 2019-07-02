import tensorflow as tf
import re
import sys
import os
from shutil import copyfile

if len(sys.argv) != 2:
    print("Usage: zero_out_chkpt_mask.py [pretrain-dir]")
    print("Creates an output directory = [pretrain-dir]_zero_mask")
    exit()
else:
    CHECKPOINT_DIR = sys.argv[1].rstrip('/')
    OUTPUT_DIR = CHECKPOINT_DIR + "_zero_mask"

os.mkdir(OUTPUT_DIR)

with tf.Session() as sess:

    # Load all the variables from the checkpoint, updating as we go
    for var_name, _ in tf.train.list_variables(CHECKPOINT_DIR):
        var_tensor_np = tf.contrib.framework.load_variable(CHECKPOINT_DIR, var_name)

        # Zero out every mask tensor
        if 'mask' == var_name[-4:]:
            var_tensor_np[:] = 0

        var = tf.Variable(var_tensor_np, name=var_name)

    # Save these new variables
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, os.path.join(OUTPUT_DIR, 'zero_mask.ckpt'))
