_base_ = [
    '../retinanet/retinanet_r50_fpn_2x_coco.py'
]

# no mask:
# distill_feat_weight=0.005
# distill_cls_weight=0.02 --> 10.

model = dict(
    type='Distilling_Base_Foreground',

    distill = dict(
        teacher_cfg='./configs/retinanet/retinanet_r101_fpn_1x_coco.py',
        teacher_model_path='./minimodel/tea_retinanet_101_mini10.pth',
        
        distill_warm_step=500,
        # distill_feat_weight=1.,
        distill_feat_weight=0.005,  # frs
        distill_cls_weight=10.,

        distill_fpn_type='frs', # foreground / frs / valid
        distill_logits_type='None', # foreground / frs / None
        
        stu_feature_adap=dict(
            type='ADAP',
            in_channels=256,
            out_channels=256,
            num=5,
            kernel=3
        ),
    )
)
data = dict(
    train=dict(
        ann_file='/data/Datasets/miniCOCO/annotations/instances_10_train2017.json',
    )
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')  # 打开tensorboard
    ])
# lr_config = dict(
#     warmup_iters=500)
custom_imports = dict(imports=['mmdet.core.utils.increase_hook'], allow_failed_imports=False)
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='Increase_Hook',)]

seed=520
# find_unused_parameters=True
