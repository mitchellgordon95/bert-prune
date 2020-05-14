task maybe_remove_masks : chkpt_utils
< in_model=$out_model@prune_pretrain
> out_model
:: remove_masks=(RemoveMasks: No Yes)
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