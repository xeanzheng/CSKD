# Improving-Knowledge-Distillation-via-Category-Structure
Codes and more experiments details for ECCV2020 paper "Improving Knowledge Distillation via Category Structure" will be released soon. We will update follow-up works in this repository.
# train CSKD
Run train_cifar.py for training CSKD in CIFAR datasets. 

`python train_cifar.py`

<br/>

Pretrained teacher model Resnet101 has been released (v1.0). 
Baidu Netdisk link is also provided:<br/>
link: https://pan.baidu.com/s/1ew_FqygwS-sRXwUS77yzag<br/>
pwd: 51yi 


<br/>

Supplementary results on Inceptionv3, Mobilenetv2, and Vgg11 in CIFAR10 are shown in following table.
| Teacher |	Student |	CE |	KD |	Fitnet |	AB |	CCKD |	Proposed |	Teacher |
|-|-|-|-|-|-|-|-|-|
| Inceptionv3 |	Mobilenetv2|	91.44|	92.03|	92.17	|92.85|	92.29	|**93.06**	|94.21|
| Inceptionv3|	Vgg11|	89.66	|92.97	|92.63	|93.38|	92.96|**93.80**	|94.21|


Supplementary results on CUB200-2011.
|Teacher|	Student	|CE	|KD	|Fitnet	|AB	|CCKD|	proposed|	Teacher|
|-|-|-|-|-|-|-|-|-|
|ResNet50|	ResNet18_0.5|	51.55|	51.81|	52.67|	53.38|	53.04|	**53.66**|	53.09|
|ResNet50|	ResNet18_0.25|	45.91|	51.60|	52.83|	52.10	|52.42|	**53.13**|	53.09|

<br/>

If this code is useful, please cite this paper.

@inproceedings{chen2020improving,

  title={Improving Knowledge Distillation via Category Structure},
  
  author={Chen, Zailiang and Zheng, Xianxian and Shen, Hailan and Zeng, Ziyang and Zhou, Yukun and Zhao, Rongchang},
  
  booktitle={European Conference on Computer Vision},
  
  pages={205--219},
  
  year={2020},
  
  organization={Springer}
  
}

<br/>
