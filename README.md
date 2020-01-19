# Bayesian Learning of Neural Network Architectures

**Disclaimer:**
This repository contains a reimplementation of the methods presented in 
[this paper](http://proceedings.mlr.press/v89/dikov19a/dikov19a.pdf).
This is not the original code used in the experimental section of the paper.

For a high-level overview of the approach refer to the following 
[blog post](https://argmax.ai/blog/archopt/).

## Project structure

The `bal` package contains a Pytorch implementation of the Bayesian adaptive size and 
skip connection layers, a custom implementation of a truncated normal distribution by 
subclassing `torch.Distribution`, and some other utilities.

The `examples` folder contains notebooks and utility functions to visually demonstrate 
the approach described in the paper and the blog post mentioned above.

## Citation
```
@InProceedings{pmlr-v89-dikov19a,
  title     = 	 {Bayesian Learning of Neural Network Architectures},
  author    = 	 {Dikov, Georgi and Bayer, Justin},
  booktitle = 	 {Proceedings of Machine Learning Research},
  pages     = 	 {730--738},
  year      = 	 {2019},
  volume    = 	 {89},
  month     = 	 {16--18 Apr}
}
```
