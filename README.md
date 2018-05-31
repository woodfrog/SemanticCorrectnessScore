# Semantic Correctness Score


This repository implements the evaluation metric **Semantic Correctness Score** proposed in the paper **Probabilistic Neural Programmed Networks for Scene Generation**. This metric measures the quality of generated images based on semantic correctness, i.e. it measures whether an image contains correct number of objects, objects with specified class, and objects with specified attributes and properties. 

Semantic Correctness Score is a **detector-based** metric, and this repo is built on the basis of the popular Pytorch faster-rcnn implementation [link](https://github.com/jwyang/faster-rcnn.pytorch).
