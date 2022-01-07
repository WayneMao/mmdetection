from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
from torch import nn
import torch.nn.functional as F
from ..losses.utils import weight_reduce_loss
# from mmdet.apis.inference import init_detector

@DETECTORS.register_module()
class Distilling_Single_QFocal_Label(SingleStageDetector):
    '''
    Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection

    Reference https://github.com/implus/GFocal
    NOTE: 这里只在logits 加QFocal
          cpu瞬时负载较高，建议--cpu 32
    '''

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_Single_QFocal_Label, self).__init__(backbone, neck, bbox_head, train_cfg,
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

        self.distill_fpn_type = distill.get('distill_fpn_type', None)
        self.num_classes = 80
        self.feat_channels = 256
        self.num_anchor = 9
        self.distill_qfl_type = distill.get('distill_qfl_type', None) # mean sum
        self.qfocal_loss = QualityFocalLoss(reduction=self.distill_qfl_type)
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        x = self.extract_feat(img)
        # losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore)
        losses, cls_reg_targets = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, teacher=True)  
        # losses{'loss_cls', 'loss_bbox'}  list[5]
        # --------------------------- recall label target ------------------------- #                     
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets                                              
        
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
            stu_cls_score_ = stu_cls_score[layer]
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values
            mask = mask.detach()

            #----------------- foreground ------------------- #
            labels = labels_list[layer].flatten() # [2*9*h*w] anchor(K, A, 4)
            valid_idxs = labels >= 0
            num_valid = valid_idxs.sum()
            foreground_idxs = (labels >=0) & (labels != self.num_classes)  # [2*9*h*w]
            num_foreground = foreground_idxs.sum()

            # ------------------- cls_loss ------------------------------ #
            # logits + Option: valid / foreground / frs / None
            stu_cls_score_sigmoid = stu_cls_score_sigmoid.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # [2, 9*80, h, w] --> [2*h*w*num_anchor, 80]
            stu_cls_score_ = stu_cls_score_.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            tea_cls_score_sigmoid = tea_cls_score_sigmoid.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            # QFocal loss
            if self.distill_qfl_type == 'mean':
                cls_loss = self.qfocal_loss(stu_cls_score_, labels, tea_cls_score_sigmoid)  # defalt mean
            elif self.distill_qfl_type == 'sum':
                # print(tea_cls_score_sigmoid.sum())
                cls_loss = self.qfocal_loss(stu_cls_score_, labels, tea_cls_score_sigmoid) / tea_cls_score_sigmoid.sum()
                if cls_loss > 1000:
                    print(cls_loss)
            
            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)

            distill_feat_loss += feat_loss.mean()
            distill_cls_loss += cls_loss

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

#@LOSSES.register_module
class QualityFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                score,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                score,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
    
def quality_focal_loss(
          pred,          # (n, 80) 未sigmoid
          label,         # (n) 0, 1-80: 80 is neg, 0-79 is positive todo
          score,         # (n, 80) reg target 0-1, only positive is good todo
          weight=None,
          beta=2.0,
          reduction='mean',
          avg_factor=None):
    # all goes to 0
    pred_sigmoid = pred.sigmoid()
    #pred_sigmoid = pred
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pred.shape) 
    loss = F.binary_cross_entropy_with_logits(
           pred, zerolabel, reduction='none') * pt.pow(beta)   # [2*9*h*w, 80]
    # todo 是否太hard 

    # import pdb
    # pdb.set_trace()
    # label = label - 1
    # pos = (label >= 0).nonzero().squeeze(1)
    num_class = pred.size(1)
    pos = ((label >= 0) & (label < num_class)).nonzero().squeeze(1)  # todo torch.__version__
    # a = pos
    pos_label = label[pos].long()
    
    # positive goes to bbox quality
    pt = score[pos, pos_label] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
           pred[pos, pos_label], score[pos, pos_label], reduction='none') * pt.pow(beta)

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss
    # TODO 正负样本各自取分母 score[pos, pos_label].sum()

# copy from https://github.com/implus/GFocal 
def quality_focal_loss_paper(
          pred,          # (n, 80)
          label,         # (n) 0, 1-80: 0 is neg, 1-80 is positive
          score,         # (n) reg target 0-1, only positive is good
          weight=None,
          beta=2.0,
          reduction='mean',
          avg_factor=None):
    # all goes to 0
    pred_sigmoid = pred.sigmoid()
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
           pred, zerolabel, reduction='none') * pt.pow(beta)

    label = label - 1
    pos = (label >= 0).nonzero().squeeze(1)
    a = pos
    b = label[pos].long()
    
    # positive goes to bbox quality
    pt = score[a] - pred_sigmoid[a, b]
    loss[a,b] = F.binary_cross_entropy_with_logits(
           pred[a,b], score[a], reduction='none') * pt.pow(beta)

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss