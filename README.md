# StyleKD
This repo is the implementation of paper [Mind the Gap in Distilling StyleGANs (ECCV2022)](https://arxiv.org/abs/2208.08840).

<img src="https://github.com/xuguodong03/StyleKD/raw/master/images/teaser.png" width="100%" height="100%">

## Running 

This is unstable version. The code is still in testing.
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=8001 distributed_train.py --name batch16_run1 \
    --load 0 --load_style 1 --g_step 0 \
    --kd_l1_lambda 3 --kd_lpips_lambda 3 --kd_simi_lambda 30 --kd_style_lambda 0 \
    --lr_mlp 0.01 --fix_w 0 --batch 4 --worker 4 \
    --simi_loss kl --single_view 0 --offset_mode main --main_direction split --offset_weight 5.0
```

## Results

### Quantitative Results
<img src="https://github.com/xuguodong03/StyleKD/raw/master/images/result.jpeg" width="100%" height="100%">

### Qualitative Results

#### Face
<img src="https://github.com/xuguodong03/StyleKD/raw/master/images/compare_1024_char.jpg" width="100%" height="100%">

#### Church
<img src="https://github.com/xuguodong03/StyleKD/raw/master/images/church.jpg" width="100%" height="100%">

#### Face Editing
<img src="https://github.com/xuguodong03/StyleKD/raw/master/images/editing_a.jpg" width="100%" height="100%">

## Citation
If you find this repo useful for your research, please consider citing the paper
```
@misc{xu2022stylekd,
  url = {https://arxiv.org/abs/2208.08840},
  author = {Xu, Guodong and Hou, Yuenan and Liu, Ziwei and Loy, Chen Change},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Mind the Gap in Distilling StyleGANs},
  publisher = {arXiv},
  year = {2022}
}

```
