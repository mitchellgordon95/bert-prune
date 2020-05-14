import "tapes/submitters.tape"
import "tapes/versioners.tape"
import "tapes/link_data.tape"
import "tapes/data_prep.tape"
import "tapes/pretraining.tape"
import "tapes/downstream.tape"

plan data_prep {
  reach train_test_split via (Chunk:*)
}

plan compare_features {
  reach compare_features via (Sparsity: 0 30) * (PretrainPrune: one_shot) 
}

plan glue {
  reach train_glue via (Sparsity: 0 30 60 90) * (PretrainPrune: one_shot) * (GlueTask: *)
  reach train_glue via (Sparsity: 30 60 90) * (PretrainPrune: random) * (GlueTask: *)
}


global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}