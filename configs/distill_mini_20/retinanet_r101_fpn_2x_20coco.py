_base_ = '../retinanet/retinanet_r50_fpn_2x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

data = dict(
    train=dict(
        ann_file='/data/Datasets/miniCOCO/annotations/instances_20_train2017.json',
    )
)        
# https://github.com/open-mmlab/mmdetection/issues/6914