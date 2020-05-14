task prunable_model : chkpt_utils
< original=$model@bert_model
> model
:: pyenv=@ {
  python $chkpt_utils/make_pretrain_chkpt_prunable.py $original $model
}

task burn_in : bert
< in_model=$model@prunable_model
< train=$train@train_test_split
< dev=$dev@train_test_split
> out_model
:: pyenv=@ .submitter=@ .resource_flags=$resource_flags_titan .action_flags=@ {

    params="--train_batch_size 32
    --max_seq_length 128
    --max_predictions_per_seq 20
    --num_warmup_steps 10
    --keep_checkpoint_max 1
    --learning_rate 2e-5
    --output_dir $out_model
    --bert_config_file $in_model/bert_config.json
    --init_checkpoint $in_model/bert_model.ckpt"

   # Disable gradient checkpointing b/c it doesn't work for pre-training
   export DISABLE_GRAD_CHECKPOINT=True

    # Burn in the model first, without pruning
    for step in $(seq 1 3); do
      python $bert/run_pretraining.py --do_train=True --num_train_steps=$(expr $step \* 5000) --input_file=$train/* $params
      python $bert/run_pretraining.py --do_eval=True --max_eval_steps=2000 --input_file=$dev/* $params
    done
}

task prune_pretrain : bert : chkpt_utils
< in_model=$out_model@burn_in
< orig_model=$model@bert_model
< train=$train@train_test_split
< dev=$dev@train_test_split
> out_model
:: sparsity=(Sparsity: 0 10 20 30 40 50 60 70 80 90)
:: prune_type=(PretrainPrune: one_shot gradual random head)
:: pyenv=@ .submitter=@ .resource_flags=$resource_flags_titan .action_flags=@ {

    # TODO we will eventually have some kind of weight_sparsity_map here
    if [ $prune_type == none ]; then
      sparsity_args=""
    elif [ $prune_type == gradual ]; then
      sparsity_args="--pruning_hparams=initial_sparsity=0,target_sparsity=.$sparsity,"
      sparsity_args+="sparsity_function_end_step=10000,end_pruning_step=-1"
    elif [ $prune_type == one_shot ]; then
      sparsity_args="--pruning_hparams=initial_sparsity=.$sparsity,target_sparsity=.$sparsity,"
      sparsity_args+="sparsity_function_end_step=1,end_pruning_step=200"
    elif [ $prune_type == random ]; then
      # Random prune with a utility, don't prune with the built-in pruning library
      sparsity_args=""
      python $chkpt_utils/random_masks.py $in_model random_pruned 0.$sparsity
      # Initialize the pre-training model with the randomly pruned thing instead
      in_model=random_pruned
    elif [ $prune_type ==  head ]; then
      # Prune attention heads with a utility, don't prune with the built-in pruning library
      sparsity_args=""
      python $chkpt_utils/random_prune.py $in_model head_pruned 0.$sparsity
      # Initialize the pre-training model with the randomly pruned thing instead
      in_model=head_pruned
    else
      echo "Invalid prune type: $prune_type"
      exit 1
    fi

    # TODO: this might be broken, if we need to use in_model/name.chkpt instead
    params="--train_batch_size 32
    --max_seq_length 128
    --max_predictions_per_seq 20
    --num_warmup_steps 10
    --keep_checkpoint_max 1
    --learning_rate 2e-5
    --output_dir $out_model
    --bert_config_file $orig_model/bert_config.json
    --init_checkpoint $in_model"

   # Disable gradient checkpointing b/c it doesn't work for pre-training
   export DISABLE_GRAD_CHECKPOINT=True

    for step in $(seq 1 20); do
        python $bert/run_pretraining.py --do_train=True --num_train_steps=$(expr $step \* 5000) --input_file=$train/* $params $sparsity_args
        # Note: max_eval_steps is the number of batches we process
        # default batch size is 8
        python $bert/run_pretraining.py --do_eval=True --max_eval_steps=2000 --input_file=$dev/* $params $sparsity_args
   done
}