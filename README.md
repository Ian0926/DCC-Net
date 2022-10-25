# Deep Color Consistent Network for Low Light-Image Enhancement (CVPR 2022)

> Pytorch implementation of [Deep Color Consistent Network for Low Light-Image Enhancement](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Deep_Color_Consistent_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf).
**Note that we do some adjustments to the model for flexibility**. Please refer to the code for more detailes.

### Environment:
> 1. Python 3.6 
> 2. Pytorch 1.3.0
> 3. torchvision 0.4.1
> 4. cuda 11.4

### Pretrained Model and Testing Results: 
> 1) Download **the pretrained model on LOL dataset** from [this link](https://drive.google.com/u/0/uc?id=134wM6wz0GdC6QXaeyrtQy6tyHpuRZ8Jp&export=download). 
> 2) **The testing results of LOL dataset** can be found [here](https://github.com/Ian0926/DCC-Net/tree/main/results).

### Test:
`python test.py --filepath img_path --pretrain_path model_path`

### Evaluation
`python metrics.py --data-path gt_path --output-path pre_img_path`

### Bibtex:
```
@inproceedings{zhang2022deep,
  title={Deep Color Consistent Network for Low-Light Image Enhancement},
  author={Zhao Zhang, Huan Zheng, Richang Hong, Mingliang Xu, Shuicheng Yan and Meng Wang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
