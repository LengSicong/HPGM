# Intelligent Home 3D: Automatic 3D-House Design from Linguistic Descriptions Only

Pytorch implementation for “Intelligent Home 3D: Automatic 3D-House Design from Linguistic Descriptions Only”

In this paper, we formulate home design as a language conditioned visual content generation problem that is further divided into a floor plan generation and an interior texture (such as floor and wall) synthesis task. The only control signal of the generation process is the linguistic expression given by users that describe the house details. To this end, we propose a House Plan Generative Model (HPGM).

<p align="center">
<img src="./images/sample.png" alt="example" width="40%">
</p>
<p align="center">
Figure: An example of generated 3D house with description using HPGM on the Text--to--3D House Model dataset.
</p>


## Framework

The proposed HPGM consists of five components: 
1. text representation block
2. graph conditioned layout prediction network (GC-LPN)
3. floor plan post-processing
4. language conditioned texture GAN (LCT-GAN)
5. 3D scene generation and rendering


<p align="center">
<img src="./images/framework.png" alt="framework" width="80%">
</p>
<p align="center">
Figure: The overview framework of HPGM.
</p>


## Dependencies

```
Python==2.7, PyTorch==0.4.1, torchvision==0.2.1
```

## Dataset

In our paper, to train and evaluate our model, we build the first Text--to--3D House Model dataset.
The dataset contains 2,000 houses, 13,478 rooms and 873 (some rooms have same textures so this number is smaller than the total number of rooms.) texture images with corresponding natural language descriptions. These descriptions are firstly generated from some pre-defined templates and then refined by human workers. The average length of the description is 173.73 and there are 193 unique words.
In our experiments, we use 1,600 pairs for training while 400 for testing in the building layout generation. For texture synthesis, we use 503 data for training and 370 data for testing.

- Download Text--to--3D House Model dataset:

1. [Text--to--3D House Model (Baidu Yun)](https://pan.baidu.com/s/1C88zjcYIAKRiqVD5Qlj7yQ) Password: n7ty

2. [Text--to--3D House Model (Google Drive)](https://pan.baidu.com/s/1C88zjcYIAKRiqVD5Qlj7yQ)


<p align="center">
<img src="./images/sample_dataset.png" alt="dataset" height="230"><img src="./images/word_cloud.png" alt="word cloud" height="230">
</p>
<p align="center">
Figure: An example from Text--to--3D House Model dataset (left) and the word cloud of the texts in our dataset (right).
</p>


## Training
- Train GC-LPN
```
python xxx.py
```

- Train LCT-GAN
```
python xxx.py
```


## Evaluation Metrics

- Calculate the IoU
```
python xxx.py
```

- Calculate the Fréchet Inception Distance (FID)
```
python xxx.fid
```

- Calculate the MS-SSIM
```
pythono xxx.py
```


## Generated examples

We provide some visualised layouts and textures results compared with baseline methods.

<p align="center">
<img src="./images/sample_GC_LPN.png" alt="GC-LPN" height="190"><img src="./images/sample_LCT_GAN.png" alt="LCT-GAN" height="190">
</p>
<p align="center">
Figure: The examples of layouts produced by GC-LPN (left) and textures generated by LCT-GAN (right).
</p>



## Citation

If you use any part of this code in your research, please cite our paper:

```
@inproceedings{chen2020intelligent,
  title={Intelligent Home 3D: Automatic 3D-House Design from Linguistic Descriptions Only},
  author={Chen, Qi and Wu, Qi and Tang, Rui and Wang, Yuhan and Wang, Shuai and Tan, Mingkui},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}

```

