This is the code for [Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning](https://arxiv.org/abs/2002.08307)

It includes /bert, which is the original BERT repository modified to be weight prunable. (And to use gradient checkpointing, if you need that.)

I am currently in the process of converting these experiments into a [ducttape](https://github.com/jhclark/ducttape) workflow, so things are a little unstable right now.

If you need *all* the experiments from the paper, check out [this commit](https://github.com/mitchellgordon95/bert-prune/commit/1fd8be2250e427e8338863feb52847c6547cd7ba). It's very messy, so be prepared to read the code. I will not be releasing a guide to run that code, since it will be made obselete by the ducttape workflow.
