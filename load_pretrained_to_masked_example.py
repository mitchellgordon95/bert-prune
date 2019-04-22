import tensorflow as tf
import modeling
import re

# Dummy training batch
input_ids = tf.constant([[31, 51, 99]])
input_mask = tf.constant([[1, 1, 1]])
token_type_ids = tf.constant([[0, 0, 1]])

CHECKPOINT = '../uncased_L-12_H-768_A-12/bert_model.ckpt'
# Bert Base
config = modeling.BertConfig(vocab_size=30522, hidden_size=768,
                             num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, type_vocab_size=2)

model = modeling.BertModel(config=config, is_training=True,
input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

var_map = {}

for var_name, _ in tf.train.list_variables(CHECKPOINT):
    new_name = re.sub(r'(query|key|value)/kernel', r'\1/weights', var_name)
    new_name = re.sub(r'(query|key|value)/bias', r'\1/biases', new_name)

    new_name = re.sub(r'dense/kernel', 'fully_connected/weights', new_name)
    new_name = re.sub(r'dense/bias', 'fully_connected/biases', new_name)

    # TODO(mitchg) - load these pre-train specific layers into the graph before loading the checkpoint.
    # Maybe also prune them? Probably not though, doesn't seem worth.
    if 'cls' not in var_name:
        var_map[var_name] = new_name

tf.train.init_from_checkpoint(CHECKPOINT, assignment_map=var_map)
summary = tf.summary.FileWriter('logdir', graph=tf.get_default_graph())
