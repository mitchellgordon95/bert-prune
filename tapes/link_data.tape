# TODO: move these to main.tconf
task link_wiki > out {
     ln -s /exp/mgordon/bert-prune/data/enwiki $out
}
task link_bookcorpus > out {
     ln -s /exp/mgordon/bert-prune/data/bookcorpus/out_txts $out
}
task bert_model > model :: model_loc=(BertSize: base="/exp/mgordon/bert-prune/uncased_L-12_H-768_A-12") {
     ln -s $model_loc $model
}
task glue_data > data_dir {
     ln -s /exp/mgordon/bert-prune/data/glue_data $data_dir

}
