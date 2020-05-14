import "tapes/submitters.tape"
import "tapes/versioners.tape"
import "tapes/link_data.tape"
import "tapes/data_prep.tape"
import "tapes/pretraining.tape"
import "tapes/downstream.tape"

plan main {
  # Prune one_shot and random pruning during pre-training, no downstream pruning
  reach prune_downstream via (Sparsity: 0 30 60 90) * (PretrainPrune: one_shot) * (GlueTask: *) * (DownstreamSparsity: 0)
  reach prune_downstream via (Sparsity: 30 60 90) * (PretrainPrune: random) * (GlueTask: *) * (DownstreamSparsity: 0)
  # Remove the masks for one-shot and keep going
  reach prune_downstream via (Sparsity: 30 60 90) * (PretrainPrune: one_shot) * (RemoveMasks: yes) * (GlueTask: *) * (DownstreamSparsity: 0)
  # Take the one-shot model with 0 pruning, prune it and continue training on downstream data
  reach prune_downstream via (Sparsity: 0) * (PretrainPrune: one_shot) * (GlueTask: *) * (DownstreamSparsity: *)
}

plan compare_features {
  reach compare_features via (Sparsity: 0 30) * (PretrainPrune: one_shot) 
}

global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}