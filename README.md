# Adversarial attack for saliency detection
**ECE 6258 final project:** <br>
Saliency detection via adversarial attack on image segmentation task

In this project, the objective focused on investigating the image saliency for a target-specific convolutional neural network on the classification task by evaluating its robustness with the presence of visually-subtle adversarial perturbations. The main goal is to reach a saliency mask generated from local perturbed noise attack to demonstrate how neural network classifies its given object.

We use two datasets for attack purpose: one is ImageNet and the other is from [DeepEyeNet](https://github.com/waynewu6250/DeepEyeNet-Keywords). First we train a ResNet50 model based on these two datasets. Then we use four different attack mechanisms to perform attack. For more information, please check the [project paper](https://github.com/waynewu6250/attack-for-saliency/blob/master/paper.pdf): 

* For DeepEyeNet dataset:
>
    python main.py perform_attack --attack-model=<mode>

* For ImageNet dataset:
>
    python main_image.py perform_attack --attack-model=<mode>

Here we could specify \<mode> to be the following:
1. `untargeted-attack`: untargeted attack
2. `targeted-attack`: targeted attack
3. `mask-targeted-attack`:
   For this mode, you should further specify the   `opt.mode` to be `partial`, `random`, `unet`

## Results
Results for 4 modes of saliency attack:


<img src="https://github.com/waynewu6250/attack-for-saliency/blob/master/imagenet.jpg">


