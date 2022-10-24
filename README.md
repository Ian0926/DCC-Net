# Deep Color Consistent Network for Low Light-Image Enhancement (CVPR 2022)

Pytorch implementation of [Deep Color Consistent Network for Low Light-Image Enhancement](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Deep_Color_Consistent_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf).

### Environment:
1. Python 3.6 
2. Pytorch 1.3.0
3. torchvision 0.4.1
4. cuda 11.4

### Pretrained Model and Visual Results: 
Download the pretrained model on LOL dataset from [this link](https://drive.google.com/u/0/uc?id=134wM6wz0GdC6QXaeyrtQy6tyHpuRZ8Jp&export=download). The results on LOL dataset can be found [here](https://github.com/Ian0926/DCC-Net/tree/main/results). Note that we have done some adjustments to the model for fewer parameters and better results.

### Test:
`python eval.py --filepath img_path --pretrain_path model_path`

### Evaluation
`python fr_metrics.py --data-path gt_path --output-path pre_img_path`

### Bibtex:
```
@inproceedings{zhang2021deep,
  title={Deep Color Consistent Network for Low-Light Image Enhancement},
  author={Zhao Zhang, Huan Zheng, Richang Hong, Mingliang Xu, Shuicheng Yan and Meng Wang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition },
  year={2022}
}
```
