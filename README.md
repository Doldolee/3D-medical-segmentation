
# 3D medical segmentation

# Dataset
- Medical Segmentation Decathlon [Link](http://medicaldecathlon.com/dataaws/)
- spleen 
- Colon 
- Heart

# Model 
- Vnet[Link](https://arxiv.org/abs/1606.04797)
- UNETR[Link](https://arxiv.org/abs/2103.10504)

# Metric (-ing)
|         |        Spleen       |         Colon       |        Heart        |  
| Model   | Dice Score | Loss   | Dice Score | Loss   | Dice Score | Loss   | 
|---------|------------|--------|------------|--------|------------|--------|
| VNet    |   0.247    | 0.961  |    0.467   | 0.712  |    0.44    |  0.78  |
| UNETR   |   0.007    | 0.992  |    0.371   | 0.865  |    0.409   |  0.79  |

# Visual
![visual](./plot/visual.png)
