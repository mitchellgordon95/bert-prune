from pretrain_experiments.common import TRAIN_128, DEV_128, pretrain, pretrain_eval

# Burn in a pre-trained model for 10k steps.
# Not sure why this is necessary, but dev loss seems to go down quite a bit.
model_name = f"burned_in"
for step in [0,1,2]:
    pretrain(
        input_file=TRAIN_128,
        model_name=model_name,
        num_train_steps=step*5000,
        sparsity_hparams=None
    )
    # TODO (mitchg) what's the right number here?
    pretrain_eval(model_name=model_name, input_file=DEV_128, max_eval_steps=2000)

