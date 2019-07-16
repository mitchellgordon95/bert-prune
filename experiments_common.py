from collections import namedtuple


class SparsityHParams(
        namedtuple('SparsityHParams', [
            'initial_sparsity',
            'target_sparsity',
            'sparsity_function_end_step',
            'end_pruning_step'])):
    """See:
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/model_pruning"""
    def __str__(self):
        return (f'initial_sparsity={self.initial_sparsity},'
                f'target_sparsity={self.target_sparsity},'
                f'sparsity_function_end_step={self.sparsity_function_end_step},'
                f'end_pruning_step={self.end_pruning_step}')