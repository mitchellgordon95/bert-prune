task sentencize : scripts 
    < wiki=$out@link_wiki < books=$out@link_bookcorpus
    > out
    :: chunk=(Chunk:0..15) pyenv=@ .submitter=@ .resource_flags=@ .action_flags=@ {
  python $scripts/preprocess_pretrain_data.py $chunk 16 $wiki $books > $out 
}

task create_pretrain_data : scripts : bert
    < sentencized=$out@sentencize
    < bert_model=$model@bert_model
    > out_dir
    :: pyenv=@ .submitter=@ .resource_flags=@ .action_flags=@ {
  mkdir $out_dir
  ln $sentencized sentencized
  python $scripts/shuffle_and_split.py sentencized 50 
  for in_file in sentencized_* ; do
    python $bert/create_pretraining_data.py \
                --input_file $in_file \
                --output_file $out_dir/$(basename $in_file) \
                --vocab_file $bert_model/vocab.txt \
                --do_lower_case True \
                --max_seq_length 128 \
                --max_predictions_per_seq 20 \
                --masked_lm_prob 0.15 \
                --random_seed 12345 \
                --dupe_factor 5
  done
}

task combine_data
< data_dirs=$out_dir@create_pretrain_data[Chunk:*]
> out {
  mkdir $out
  cur_fname=1
  for dir in $data_dirs; do
    for file in $dir/*; do
      ln -s $file $out/$cur_fname
      cur_fname=$(expr $cur_fname + 1)
    done
  done
}

task train_test_split : scripts
< data=$out@combine_data
> train
> test
> dev :: pyenv=@ {
  python $scripts/train_test_split.py $data $train $test $dev
}