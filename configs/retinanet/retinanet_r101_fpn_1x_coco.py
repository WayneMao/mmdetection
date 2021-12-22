_base_ = './retinanet_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='./model/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth')))
                      #checkpoint='torchvision://resnet101')))
