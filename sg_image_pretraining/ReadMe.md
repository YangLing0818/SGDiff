### Masked Contrastive Pre-Training of Scene Graphs and Images

1. Modify the ```<path/to/data>``` in the ```training/configs.py``` and ```training/configs_coco.py```

2. Run 

```shell
conda activate sgdiff
# for vg
python trainer.py
# for coco
python trainer_coco.py
```

