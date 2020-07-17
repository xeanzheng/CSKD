# Improving-Knowledge-Distillation-via-Category-Structure
Codes and more experiments details for ECCV2020 paper "Improving Knowledge Distillation via Category Structure" will be released soon. We will update follow-up works in this repository.

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
