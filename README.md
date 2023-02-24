# Intelligent Home 3D baseline on Graph2Plan data

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

```Python==3.7, PyTorch==1.8.1```

## Dataset

We transform the dataset of Graph2Plan to the data format of Intelligent Home 3D. We split the data following Graph2Plan as well. 


## Training
- Train GC-LPN
```
python main.py --cfg cfg/layout.yml --gpu '0'
```

## Testing
- Test GC-LPN
```
python main.py --cfg cfg/layout_test.yml --gpu '0'
```

## Citation

If you use any part of this code in your research, please cite the paper:

```
@inproceedings{chen2020intelligent,
  title={Intelligent home 3d: Automatic 3d-house design from linguistic descriptions only},
  author={Chen, Qi and Wu, Qi and Tang, Rui and Wang, Yuhan and Wang, Shuai and Tan, Mingkui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12625--12634},
  year={2020}
}

```

