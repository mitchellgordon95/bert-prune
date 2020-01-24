import "tapes/submitters.tape"
import "tapes/versioners.tape"
import "tapes/data_prep.tape"

plan data_prep {
  reach train_test_split via (Chunk:*)
}

global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}