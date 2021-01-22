# Evaluating Capability of Deep Neural Networks for Image Classification via Information Plane
This code is a PyTorch implementation of our paper "[Evaluating Capability of Deep Neural Networks for Image Classification via Information Plane](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hao_Cheng_Evaluating_Capability_of_ECCV_2018_paper.pdf)" accepted by ECCV2018.

The code is run on the Tesla V-100.
## Prerequisites
Python 3.6.9

PyTorch 1.2.0

Torchvision 0.5.0


## Steps on Runing IB on CIFAR 10

```
python experiment/cnn_model_cifar10_train.py
```


## Citation

If you find this code useful, please cite the following paper:

```
@inproceedings{cheng2018evaluating,
  title={Evaluating capability of deep neural networks for image classification via information plane},
  author={Cheng, Hao and Lian, Dongze and Gao, Shenghua and Geng, Yanlin},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={168--182},
  year={2018}
}
```






