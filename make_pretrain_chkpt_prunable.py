import tensorflow as tf
import modeling
import re
import sys
import os
from shutil import copyfile

if len(sys.argv) < 2:
    print("Usage: make_pretrain_chkpt_prunable.py [pretrain-dir]")
    print("Creates an output directory = [pretrain-dir]_prunable")
    exit()
else:
    CHECKPOINT_DIR = sys.argv[1].rstrip('/')
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'bert_model.ckpt')
    OUTPUT_DIR = CHECKPOINT_DIR + "_prunable"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'bert_model.ckpt')

os.mkdir(OUTPUT_DIR)

# Copy all the other files that aren't checkpoint related too
for fname in os.listdir(CHECKPOINT_DIR):
    if 'ckpt' not in fname:
        copyfile(os.path.join(CHECKPOINT_DIR, fname), os.path.join(OUTPUT_DIR,fname))

with tf.Session() as sess:

    # Load all the variables from the checkpoint, renaming as we go
    for var_name, _ in tf.train.list_variables(CHECKPOINT_FILE):
        var_tensor = tf.contrib.framework.load_variable(CHECKPOINT_FILE, var_name)

        # Note: we don't prune the last MLP before the output layer, since
        # it isn't used after pre-training, and it's probably not going to prune much
        # (since later layers tend to prune way less)
        if 'cls' in var_name:
            # To find this layer in the code, grep for cls/predictions
            new_name = var_name
        else:
            new_name = re.sub(r'(query|key|value)/kernel', r'\1/weights', var_name)
            new_name = re.sub(r'(query|key|value)/bias', r'\1/biases', new_name)

            new_name = re.sub(r'dense/kernel', 'fully_connected/weights', new_name)
            new_name = re.sub(r'dense/bias', 'fully_connected/biases', new_name)

        if new_name != var_name:
            print(f"Renaming {var_name} to {new_name}")

        var = tf.Variable(var_tensor, name=new_name)

    # Save these new variables
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, OUTPUT_FILE)
