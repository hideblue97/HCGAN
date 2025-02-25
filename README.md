# HCGAN （Haze-Cycle GAN）
Included here are papers `Unpaired Image-to-image Translation for Paired Hazy Image Synthesis` part of the work and the `SMOKE` dataset and the `FO-Haze` dataset.  
## Abstract
We propose the `HCGAN` model, modified from the `CycleDehaze`. It is used to synthesise paired dehazing datasets that can be used for training and validation from unpaired smoke image and clear image datasets.  
## Environment:
- CUDA Version: 12.4 
- Python 3.11.5
- torch==2.3.0
- torchvision==0.18.0
## Test & Train
```
python test8.py --dataroot datasets/smoke/ --cuda
python train8.py --dataroot datasets/smoke/ --cuda
python -m visdom.server
```
You need to start visdom for visualisation before training.
## Datasets
In our work, two datasets are presented.
### SMOKE
The smoke dataset was inspired by the [FLAME2](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset)
dataset, a collection of side-by-side infrared and visible spectral video pairs captured by an unmanned aerial vehicle (UAV)
during an open-canopy prescribed fire in northern Arizona in
2021. We have cut out 2,088 haze-free images
and 2,303 hazy images in 256*256 size from FLAME2 dataset.  

<div style="text-align: center">
<img alt="" src="images/smokeDataset.jpg" style="display: inline-block;" />
</div>  

Download address `https://pan.baidu.com/s/1kas1RcjRx4q-fEHL_S0gZQ?pwd=gbib`  
Password : `gbib`

### Fo-HAZE
`Fo-HAZE` is a pairwise dehazing dataset generated from SMOKE dataset after HCGAN model. It contains 1888 pairs of images for training and 200 pairs of images for testing.


<div style="text-align: center">
<img alt="" src="images/image1.jpg" style="display: inline-block;" />
</div>  

Download address `https://pan.baidu.com/s/1MGYt9mv1iPmaPlWqLo0tRA?pwd=gys7`

Password : `gys7` 

## Acknowledgement
We thank the authors of [CycleDehaze](https://github.com/engindeniz/Cycle-Dehaze), [ResidualDenseNet](https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch). Part of our code is built upon their modules.
