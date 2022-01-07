_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

data = dict(
    train=dict(
        ann_file='/data/Datasets/miniCOCO/annotations/instances_20_train2017.json',
    )
)