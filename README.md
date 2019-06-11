# Semantic Correctness Score

This repository implements the evaluation metric **Semantic Correctness Score** proposed in the paper **Probabilistic Neural Programmed Networks for Scene Generation**. This metric measures the quality of generated images based on semantic correctness, i.e. it measures whether an image contains a correct number of objects, objects with specified class, and objects with specified attributes and properties. 

Semantic Correctness Score is a **detector-based** metric, and this repo is built on the basis of the popular Pytorch faster-rcnn implementation [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).  

**Note: Check the original Faster-rcnn repo(the link above) for preparation and initial setup including compiling necessary files**

## Usage 
### Train the detector

To train a detector on a given dataset:

```shell
    CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                       --dataset DATASET_NAME_WITH_DETECTOR_TYPE --net res101 \
                       --bs 1 --nw 0 \
                       --lr 0.001 --lr_decay_step 5 \
                       --cuda 
```

or set all command line arguments in train_script.sh and simply run:

```shell
    sh train_script.sh
```


### Get the semantic correctness score

Our pre-trained weights for CLEVR 64X64: [link](https://www.dropbox.com/s/b7iuvxucp5lo2mz/detectors.zip?dl=0)

Get the semantic correctness score by running a test using a trained detector:

```shell
    CUDA_VISIBLE_DEVICES=0 python test_net.py \
                       --dataset DATASET_NAME_WITH_DETECTOR_TYPE --net res101 \
                       --checksession 1 --checkepoch 19 --checkpoint 4999 \
                       --cuda

```

The command line arguments should be set properly according to the actual use case. Or set them in test_script.sh and run:

```shell
    sh test_script.sh
```


## Concepts

1. **Objectness**

    A detector is firstly trained on the training set to detect all objects in the image. Ideally, the trained detector's mAP should be almost 100% on the ground-truth test data. When testing the performance of models which generate complex scenes, the objectness score is a simplified mAP based on the matching of the number of objects (without considering bounding box location).
    
    A good objectness score means that the generative model learns to generate object-like stuff and the generation is consistent with given semantics on the number of objects.

2. **Object type**
    
    Similar to the objectness score, but for object type score, the detector is trained to also classify the type of objects.
    
    Object type score is one level higher than the objectness score. A generative model will get a good object type score only when it successfully produces a correct number of objects with the correct class.
    
3. **Object-attribute combination**

    Similar to the above two, but for object-attributes combinations. The model has to generate objects matching all semantic information to get a good score. For example, in CLEVR-G dataset, objects are described by their class and a set of attributes, e.g. a large red metal sphere and a small blue rubber cube. Then a good generative model should generate exactly such objects with these attributes. 

4. **Relationships between objects**

    The semantic correctness score does not measure the correctness of relationships between objects in the scene since the ground-truth bounding boxes only give one of all possibilities of the given relationship and there can be lots of positions for generated objects to satisfy the relationship. Other metrics are required if we need to measure the relationships in generated scenes.
    
    


## To-do

- add the instructions on how to use supported dataset and detector types


## Citation
If you find the codes useful for your work, please cite the paper below:
```
@incollection{NIPS2018_7658,
title = {Probabilistic Neural Programmed Networks for Scene Generation},
author = {Deng, Zhiwei and Chen, Jiacheng and FU, YIFANG and Mori, Greg},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {4028--4038},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7658-probabilistic-neural-programmed-networks-for-scene-generation.pdf}
}

```
