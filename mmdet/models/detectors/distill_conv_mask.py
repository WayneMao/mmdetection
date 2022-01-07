from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmdet.apis.inference import init_detector

@DETECTORS.register_module()
class Distilling_Conv_Mask(SingleStageDetector):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_Conv_Mask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained)
        from mmdet.apis.inference import init_detector

        self.device = torch.cuda.current_device()
        self.teacher = init_detector(distill.teacher_cfg, \
                        distill.teacher_model_path, self.device)
        self.stu_feature_adap = build_neck(distill.stu_feature_adap)

        self.distill_feat_weight = distill.get("distill_feat_weight",0)
        self.distill_cls_weight = distill.get("distill_cls_weight",0)

        for m in self.teacher.modules():
            for param in m.parameters():
                param.requires_grad = False
        self.distill_warm_step = distill.distill_warm_step
        self.debug = distill.get("debug",False)
        self.mask_conv = nn.ModuleList()
        num_layers = 5
        anchor_based_input = 720  # 9*80
        output_channels = 80
        for _ in range(num_layers):
            self.mask_conv.append(
                nn.Sequential(
                nn.Conv2d(anchor_based_input, output_channels, 3, padding=(1, 1)),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, 3, padding=(1, 1)),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, 1, 3, padding=(1, 1)),
                nn.BatchNorm2d(1),
                nn.ReLU(),
            ))
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        stu_feature_adap = self.stu_feature_adap(x)
        y = self.teacher.extract_feat(img)

        stu_bbox_outs = self.bbox_head(x)
        stu_cls_score = stu_bbox_outs[0]

        tea_bbox_outs = self.teacher.bbox_head(y)
        tea_cls_score = tea_bbox_outs[0]

        layers = len(stu_cls_score)
        distill_feat_loss, distill_cls_loss = 0, 0

        for layer in range(layers):
            stu_cls_score_sigmoid = stu_cls_score[layer].sigmoid()
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()  # [2,720,100,152]
            # ---------------- conv mask ------------------------ #
            stu_cls_score[layer] = stu_cls_score[layer].detach()
            tea_cls_score[layer] = tea_cls_score[layer].detach()
            diff = tea_cls_score[layer] - stu_cls_score[layer]
            mask = self.mask_conv[layer](diff)   # [n,1,h,w]
            mask = mask.sigmoid()

            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='none')

            distill_feat_loss += (feat_loss * mask).sum() / mask.sum()
            distill_cls_loss +=  (cls_loss * mask).sum() / mask.sum()  # [:,None,:,:]

        distill_feat_loss = distill_feat_loss * self.distill_feat_weight
        distill_cls_loss = distill_cls_loss * self.distill_cls_weight

        if self.debug:
            # if self._inner_iter == 10:
            #     breakpoint()
            print(self._inner_iter, distill_feat_loss, distill_cls_loss)

        if self.distill_warm_step > self.iter:
            distill_feat_loss = (self.iter / self.distill_warm_step) * distill_feat_loss
            distill_cls_loss = (self.iter / self.distill_warm_step) * distill_cls_loss

        if self.distill_feat_weight:
            losses.update({"distill_feat_loss":distill_feat_loss})
        if self.distill_cls_weight:
            losses.update({"distill_cls_loss":distill_cls_loss})

        return losses
