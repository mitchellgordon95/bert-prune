import tensorflow as tf
import numpy as np
import os
import sys

if len(sys.argv) != 2:
    print("Usage: permute_masks.py [pretrain-dir]")
    print("Randomly perumutes the values of each mask matrix. Only saves mask values.")
    print("Creates directory [pretrain-dir]_randomized_mask")
    exit()
else:
    CHECKPOINT_DIR = sys.argv[1].rstrip('/')
    OUTPUT_DIR = CHECKPOINT_DIR + "_randomized_mask"

np.random.seed(1)

os.mkdir(OUTPUT_DIR)

with tf.Session() as sess:

    # Load all the variables from the checkpoint
    for var_name, _ in tf.train.list_variables(CHECKPOINT_DIR):
        # Only save the ones that end in /mask
        if var_name.endswith('/mask'):
            var_tensor_np = tf.contrib.framework.load_variable(CHECKPOINT_DIR, var_name)

            # Convert to int64 before shuffling. There's some weirdness with numpy shuffling...
            # the sums before and after don't match.
            orig_sum = np.sum(var_tensor_np > 0)
            np.random.shuffle(var_tensor_np)
            try:
                assert np.sum(var_tensor_np > 0) == orig_sum
            except:
                import pdb; pdb.set_trace()

            var = tf.Variable(var_tensor_np, name=var_name)

    # Save these new variables
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, os.path.join(OUTPUT_DIR, 'no_mask.ckpt'))
