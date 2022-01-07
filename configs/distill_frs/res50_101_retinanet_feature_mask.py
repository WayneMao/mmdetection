_base_ = [
    '../retinanet/retinanet_r50_fpn_2x_coco.py'
]

model = dict(
    type='Distilling_Feature_Mask',

    distill = dict(
        teacher_cfg='./configs/retinanet/retinanet_r101_fpn_1x_coco.py',
        teacher_model_path='./model/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth',
        
        distill_warm_step=500,
        distill_feat_weight=0.005,
        distill_cls_weight=0.02,
        
        stu_feature_adap=dict(
            type='ADAP',
            in_channels=256,
            out_channels=256,
            num=5,
            kernel=3
        ),
    )
)
# log_config = dict(
#     interval=1,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(type='TensorboardLoggerHook')
#     ])

custom_imports = dict(imports=['mmdet.core.utils.increase_hook'], allow_failed_imports=False)
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='Increase_Hook',)]

seed=520
# find_unused_parameters=True
