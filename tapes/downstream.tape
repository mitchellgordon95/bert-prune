task maybe_remove_masks : chkpt_utils
< in_model=$out_model@prune_pretrain
> out_model
:: remove_masks=(RemoveMasks: no yes)
:: pyenv=@ {
  if [ $remove_masks == Yes ]; then
    python $chkpt_utils/remove_masks.py $in_model $out_model
  else
    ln -s $in_model $out_model
  fi
}

task train_glue : bert
< in_model=$out_model@maybe_remove_masks
< orig_model=$model@bert_model
< glue_data_dir=$data_dir@glue_data
> out_model
:: glue_task=(GlueTask: CoLA SST-2 QNLI QQP MNLI)
:: pyenv=@ .submitter=@ .resource_flags=$resource_flags_2080 .action_flags=@ {

        params="--task_name $glue_task
        --data_dir $glue_data_dir/$glue_task
        --init_checkpoint $in_model
        --output_dir $out_model 
        --bert_config_file $orig_model/bert_config.json
        --vocab_file $orig_model/vocab.txt
        --train_batch_size 32
        --max_seq_length 128
        --keep_checkpoint_max 1
        --learning_rate 2e-5"

    for epoch in $(seq 1 6); do
        python $bert/run_classifier.py --do_train=True --num_train_epochs=$epoch $params
        python $bert/run_classifier.py --do_eval=True $params
   done
   python $bert/run_classifier.py --do_eval=True --eval_train_data=True $params 
}

task prune_downstream : bert : chkpt_utils
< in_model=$out_model@train_glue
< orig_model=$model@bert_model
< glue_data_dir=$data_dir@glue_data
> out_model
:: pretrain_sparsity=$sparsity@prune_pretrain
:: sparsity=(DownstreamSparsity: 0 10 20 30 40 50 60 70 80 90)
:: prune_type=(DownstreamPrune: one_shot)
:: glue_task=$glue_task@train_glue
:: pyenv=@ .submitter=@ .resource_flags=$resource_flags_2080 .action_flags=@ {

    if [ $pretrain_sparsity != 0 ] && [ $sparsity != 0 ]; then
      echo "Error: cannot prune both during pre-training and fine-tuning."
      exit 1
    fi

    if [ $prune_type == one_shot ]; then
      sparsity_args="--pruning_hparams=initial_sparsity=.$sparsity,target_sparsity=.$sparsity,"
      sparsity_args+="sparsity_function_end_step=1,end_pruning_step=200"
    else
      echo "Invalid prune type: $prune_type"
      exit 1
    fi

    # TODO: this might be broken, if we need to use in_model/name.chkpt instead
    params="--task_name $glue_task
    --data_dir $glue_data_dir/$glue_task
    --init_checkpoint $in_model
    --output_dir $out_model 
    --bert_config_file $orig_model/bert_config.json
    --vocab_file $orig_model/vocab.txt
    --train_batch_size 32
    --max_seq_length 128
    --keep_checkpoint_max 1
    --learning_rate 2e-5"

    for epoch in $(seq 1 6); do
        python $bert/run_classifier.py --do_train=True --num_train_epochs=$epoch $params $sparsity_args
        python $bert/run_classifier.py --do_eval=True $params $sparsity_args
   done
   python $bert/run_classifier.py --do_eval=True --eval_train_data=True $params 

}