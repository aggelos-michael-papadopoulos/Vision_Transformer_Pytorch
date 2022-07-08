# Vision_Transformer_Pytorch

Urls used for this repository:
* https://github.com/google-research/vision_transformer
* https://github.com/lucidrains/vit-pytorch/blob/main/README.md#vision-transformer---pytorch
* https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_%F0%9F%A4%97_Trainer.ipynb#scrollTo=LSnQ0eX0t1bd
* https://github.com/rwightman/pytorch-image-models
* https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/vision_transformer
* https://www.youtube.com/watch?v=ovB0ddFtzzA



# Overall view
Paper link: https://arxiv.org/abs/2010.11929
![Screenshot from 2022-07-08 18-03-02](https://user-images.githubusercontent.com/65830412/178019085-2bd0ea6f-bccf-4ae9-ba28-38d3ff637d14.png)

1 st: We create Vision Transformer model from scratch in "Vit.py"

2 nd: We verify that our model (which is a slight simplified version of the original==google model) has the exact parameters (i.e. is the same model) as the pretrained "vit_base_patch16_384" in "verify.py"

3 rd: We evaluate our model with our own images in "forward.py" by showing top-k (k=10) results

##  Fine tuning 
On the "fine_tune.py" we fine tune a pre-trained vision transformer on CIFAR-10 dataset (60,000 32x32 colour images in 10 classes, with 6000 images per class) achienving SOTA results. The vision transformer The model itself is pre-trained on ImageNet-21k, a dataset of 14 million labeled images.  
