# Margin Maximization for Robust Classification

This is code for "Margin Maximization for Robust Classification using Deep Learning". The idea is to maximize the decision margin which in turn improves the robustness of classifier. Unlike adversarial training, information about the dependency between model parameters and adversarial noise is not discarded which helps to improve robustness.

Links to the [paper](https://github.com/aam-at/margin_maximization/blob/master/paper/paper.pdf) and [slides](https://github.com/aam-at/margin_maximization/blob/master/paper/slides.pdf).


# Requirements:
 - Theano
 - Lasagne
 - [tensorboard](https://github.com/dmlc/tensorboard)

# Training

To reproduce `margin_maximization` results run `train_margin.py --lmbd 0.1`. For other method (`standard`, `at`, `vat`), you can run scripts `train_standard.py`, `train_at.py`, `train_vat.py` with standard parameters. Take a note that in the paper results for `vat` were reported using full dataset (`60000` examples) while for other methods `50000` examples were used.

# Testing

To test robustness of the model with DeepFool, use `test.py --load_dir <load_path>`.

# Citation

If you find this code useful for your research, please consider citing it:
```
@InProceedings{	  matyasko2017margin,
  author	= {A. Matyasko and L. P. Chau},
  title		= {Margin maximization for robust classification using deep learning},
  booktitle	= {2017 International Joint Conference on Neural Networks (IJCNN)},
  year		= {2017},
  pages		= {300-307},
  month		= {May},
  doi		= {10.1109/IJCNN.2017.7965869}
}
```
