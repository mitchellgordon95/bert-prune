#! /bin/bash
#$ -cwd
#$ -V
#$ -l num_proc=2,h_rt=10:00:00
#$ -j y
#$ -m ase
#$ -M mitchell.gordon95@gmail.com
task download_wiki > out {
  wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 -o enwiki-latest-pages-articles.xml.bz2
  bunzip2 enwiki-latest-pages-articles.xml.bz2
  # TODO - this should be a package or something
  git clone git@github.com:attardi/wikiextractor.git data/wikiextractor
  python data/wikiextractor/WikiExtractor.py --json -o $out enwiki-latest-pages-articles.xml
}

task download_bookcorpus > out {
  ## TODO - this should be a package or something
  git clone git@github.com:soskek/bookcorpus.git bookcorpus
  pip install -r bookcorpus/requirements.txt
  python bookcorpus/download_files.py --list url_list.jsonl --out out --trash-bad-count
}

# TODO: move this branch to main.tconf
task bert_model > model :: model_loc=(BertSize: base="/exp/mgordon/bert-prune/uncased_L-12_H-768_A-12") {
     ln -s $model_loc $model
}


task create_pretrain_data
    < wiki=$out@download_wiki < books=$out@download_bookcorpus
    < bert_model=$model@bert_model
    > out_dir
    :: chunk=(Chunk: 1..8)  {
  mkdir out_dir
  python scripts/preprocess_pretrain_data.py $chunk 8 $wiki $books > sentencized
  python scripts/shuffle_and_split.py sentencized 100
  for in_file in sentencized_* ; do
    python create_pretraining_data.py \
                --input_file $in_file \
                --output_file $out_dir/$(basename $in_file) \
                --vocab_file $bert_model/vocab.txt \
                --do_lower_case True \
                --max_seq_length 128 \
                --max_predictions_per_seq 20 \
                --masked_lm_prob 0.15 \
                --random_seed 12345 \
                --dupe_factor 5 \
  done
}

task train_test_split :: data_dirs=$out_dir@create_pretrain_data[Chunk: *] > train > test > dev {
     for data_dir in $datadirs; do
         python scripts/train_test_split.py $data_dir $train $test $dev
     done
}