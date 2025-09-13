# Bridging the Semantic Gap for Zero-Shot Object Counting with Language-augmented Pseudo-exemplars

<img src="/home/lcc/Downloads/overview.jpg" style="zoom: 75%;" />

### Environment

```python
pip install -r requirements.txt
```

### Dataset

#### FSC-147

Download the FSC147 dataset as instructed in its [official repository](https://github.com/cvlab-stonybrook/LearningToCountEverything). Make sure to also download the `annotation_FSC147_384.json` and `Train_Test_Val_FSC_147.json` and place them alongside the image directory (`images_384_VarV2`) in the directory of your choice.

```python
├─FSC147/    
│  ├─gt_density_map_adaptive_384_VarV2/
│  ├─images_384_VarV2/
│  ├─ImageClasses_FSC147.txt
│  ├─Train_Test_Val_FSC_147.json
│  ├─ annotation_FSC147_384.json
```

#### CARPK

Download the CARPK dataset as instructed in its [official repository](https://lafi.github.io/LPN/). 

```python
├─data/
├─├─Annotations/
├─├─Images/
├─├─ImageSets/
```

### Preparing Pretrained MAE and CLIP model

Download the pretrained model here: [mae_pretrain_vit_base_full.pth](https://github.com/facebookresearch/mae), [ViT-B-16.pt](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)

### Train

```python
python train_val.py
```

### Test

FSC-147:

```python
python test.py
```

CARPK:

```python
python test_car.py
```

### Results

| Dataset | Val MAE | Val RMSE | Test MAE | Test RMSE |
| ------- | :-----: | :------: | :------: | :-------: |
| FSC-147 |  14.52  |  55.96   |  12.97   |   95.94   |
| CARPK   |    -    |    -     |   8.90   |   11.14   |

### Visualization

<img src="/home/lcc/Downloads/vis1.jpg" style="zoom: 33%;" />

<img src="/home/lcc/Downloads/vis2.jpg" style="zoom: 26%;" />

### Acknowledgement

This project is based on implementation from [FamNet](https://github.com/cvlab-stonybrook/LearningToCountEverything) and [CACViT](https://github.com/Xu3XiWang/CACViT-AAAI24).