This is the code for [Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning](https://arxiv.org/abs/2002.08307)

It includes /bert, which is the original BERT repository modified to be weight prunable. (And to use gradient checkpointing, if you need that. This can be disabled by setting a unix environment variable DISABLE_GRAD_CHECKPOINT=True. This only works during fine-tuning, not during pre-training.)

I am currently in the process of converting these experiments into a [ducttape](https://github.com/jhclark/ducttape) workflow, so things are a little unstable right now.

Things that have not been converted to ducttape:
- Anything in tables/
- Anything in graphs/

If you need *all* the experiments from the paper, check out [this commit](https://github.com/mitchellgordon95/bert-prune/commit/1fd8be2250e427e8338863feb52847c6547cd7ba). It's very messy, so be prepared to read the code. I will not be releasing a guide to run that code, since it will be made obselete by the ducttape workflow.

Configuration
=============

    pip install -r requirements.txt

To pre-train, you will need a GPU with at least 12 GB of GPU RAM. I've been using Titan RTX's via Univa Grid Engine. If you don't like this setup, you will need to modify tapes/submitters.tape and/or main.tconf.

You'll also need the Wikipedia corpus and BookCorpus, which can be retrieved with scripts/download_wiki.sh or scripts/download_bookcorpus.sh, respectively. GLUE data can be retrieved by running scripts/get_glue.py.

You will need to update tapes/link_data.tape to point to dataset locations.

You will also need to update main.tconf to point to the location of your repository on disk (so ducttape knows where to find packages).

AFAIK, no one besides me has used this code. If you have trouble, please open an issue and I'll do what I can to help out.

Most experiments are run using

    ducttape main.tape -C main.tconf -p main
