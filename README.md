# Semantic Correctness Score


This repository implements the evaluation metric **Semantic Correctness Score** proposed in the paper **Probabilistic Neural Programmed Networks for Scene Generation**. This metric measures the quality of generated images based on semantic correctness, i.e. it measures whether an image contains correct number of objects, objects with specified class, and objects with specified attributes and properties. 

Semantic Correctness Score is a **detector-based** metric, and this repo is built on the basis of the popular Pytorch faster-rcnn implementation [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).


## Details of Semantic Correctness Score

1. **objectness**

    A detector is firstly trained on the training set to detect all objects in the image. Ideally, the trained detector's mAP should be almost 100% on the ground-truth test data. When testing the performance of models which generate complex scenes, the objectness score is a simplified mAP based on the matching of number of objects (without consider bounding box location).
    
    A good objectness score means that the generative model learns to generate object-like stuffs and the generation is consistent with given semantics on the number of objects.

2. **object type**
    
    Similar with the objectness score, but for object type score, the detector is trained to also classify the type of objects.
    
    Object type score is one level higher than the objectness score. A generative model will get good object type score only when it successfully produces correct number of objects with correct class.
    
3. **object-attribute combination**

    Object-attribute combination score is the most difficult. The model has to generate objects matching all semantic information to get good score. For example, in CLEVR-G dataset, an object is described by its class and a set of attributes, e.g. a large red metal sphere, then a good generative model should generate exactly such a sphere with these attributes. 

## Train the detector

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


## Get the semantic correctness score

Get the semantic correctness score by running a test using trained detector:

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





## To-do


- add the instructions on how to use supported dataset and detector types