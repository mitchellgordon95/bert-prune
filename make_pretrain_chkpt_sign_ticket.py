import tensorflow as tf
import modeling
import re
import sys
import os
from shutil import copyfile

if len(sys.argv) != 3:
    print("Usage: make_pretrain_chkpt_sign_ticket.py [pretrain-dir] [stddev]")
    print("Creates an output directory = [pretrain-dir]_sign_ticket")
    print("stddev is the param used to initialize weight matrices")
    exit()
else:
    CHECKPOINT_DIR = sys.argv[1].rstrip('/')
    OUTPUT_DIR = CHECKPOINT_DIR + "_sign_ticket"
    STDDEV = float(sys.argv[2])

os.mkdir(OUTPUT_DIR)

with tf.Session() as sess:

    # Load all the variables from the checkpoint, renaming as we go
    for var_name, _ in tf.train.list_variables(CHECKPOINT_DIR):
        var_tensor = tf.contrib.framework.load_variable(CHECKPOINT_DIR, var_name)

        # Replace each value in each weights matrix with STDDEV of the same sign
        if 'weights' == var_name[-7:]:
            var_tensor[var_tensor > 0] = STDDEV
            var_tensor[var_tensor < 0] = -STDDEV

        # TODO (mitchg) - should we be keeping other variables like LayerNorm and Adam?
        var = tf.Variable(var_tensor, name=var_name)

    # Save these new variables
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, OUTPUT_DIR)
