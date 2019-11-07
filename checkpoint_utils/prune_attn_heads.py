import tensorflow as tf
import numpy as np
import re
import sys
import os
from shutil import copyfile
import fire
from checkpoint_utils.common import prune

SIZE_PER_HEAD = int(768 / 12) # This is only true for BERT-Base

def params_for_attn(ledger, layer, masks=False):
    end = 'mask' if masks else 'weights'
    return (
        ledger[f'bert/encoder/layer_{layer}/attention/self/key/{end}'],
        ledger[f'bert/encoder/layer_{layer}/attention/self/query/{end}'],
        ledger[f'bert/encoder/layer_{layer}/attention/self/value/{end}'],
        ledger[f'bert/encoder/layer_{layer}/attention/output/fully_connected/{end}'],
        )

def extract_single_head(key, query, value, FC, head_ind):
    assert all([tensor.shape == (768, 768) for tensor in [key, query, value, FC]])

    return tuple(tensor[:,SIZE_PER_HEAD*head_ind:SIZE_PER_HEAD*(head_ind+1)]
                 for tensor in [key, query, value]
                 ) + (FC[SIZE_PER_HEAD*head_ind:SIZE_PER_HEAD*(head_ind+1),:],)


def attn_head_weight(key, query, value, FC, head_ind):
    total = 0
    for tensor in extract_single_head(key, query, value, FC, head_ind):
        total += np.sum(np.abs(tensor))

    return total


def prune_single_head(ledger, layer, head):
    key, query, value, FC = params_for_attn(ledger, layer, masks=True)

    for tensor in [key, query, value]:
        tensor[:,SIZE_PER_HEAD*head:SIZE_PER_HEAD*(head+1)] = 0

    FC[SIZE_PER_HEAD*head:SIZE_PER_HEAD*(head+1),:] = 0


def prune_attn_heads(model_dir, sparsity: float):
    """Prunes [sparsity] of the number of attention heads in [model_dir].
    Makes a new checkpoint [model_dir]_head_pruned_[sparsity].
    """
    model_dir = model_dir.rstrip('/')

    with tf.Session() as sess:

        # Load all the variables from the checkpoint
        ledger = {}
        for var_name, _ in tf.train.list_variables(model_dir):
            ledger[var_name] = tf.contrib.framework.load_variable(model_dir, var_name)

        head_weights = np.zeros((12, 12)) # 12 layers, 12 heads each
        # layer_stds = np.zeros(12) # 12 layers
        for layer in range(12):
            params = params_for_attn(ledger, layer)
            # layer_stds[layer] = np.std(np.concatenate(params).flatten())
            for head in range(12):
                head_weights[layer,head] = attn_head_weight(*params, head)

        # layer_stds /= np.linalg.norm(layer_stds, keepdims=True)

        # TODO: normalize by layer
        layer_norms = np.sum(head_weights, axis=1, keepdims=True)
        # layer_norms = np.linalg.norm(head_weights, axis=1, keepdims=True)
        head_weights /= layer_norms
        # head_weights *= layer_stds
        mask = prune(head_weights, sparsity)

        # Non-zero gives us the indices of non-zero elements like
        # ([row indices], [column indices])
        to_prune = np.nonzero(mask == 0)
        for layer, head in zip(to_prune[0], to_prune[1]):
            prune_single_head(ledger, layer, head)

        for var_name, var_tensor_np in ledger.items():
            var = tf.Variable(var_tensor_np, name=var_name)

        # Save these new variables
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        output_dir = model_dir + f"_head_pruned_{int(sparsity*100)}"
        os.mkdir(output_dir)
        saver.save(sess, os.path.join(output_dir, 'head_pruned.ckpt'))

if __name__ == '__main__':
    fire.Fire(prune_attn_heads)
