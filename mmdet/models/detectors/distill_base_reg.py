from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
import torch.nn.functional as F
# from mmdet.apis.inference import init_detector

@DETECTORS.register_module()
class Distilling_Base_Reg(SingleStageDetector):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_Base_Reg, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained)
        from mmdet.apis.inference import init_detector

        self.device = torch.cuda.current_device()
        self.teacher = init_detector(distill.teacher_cfg, \
                        distill.teacher_model_path, self.device)
        self.stu_feature_adap = build_neck(distill.stu_feature_adap)

        self.distill_feat_weight = distill.get("distill_feat_weight",0)
        self.distill_cls_weight = distill.get("distill_cls_weight",0)
        self.distill_reg_weight = distill.get("distill_reg_weight",0)

        for m in self.teacher.modules():
            for param in m.parameters():
                param.requires_grad = False
        self.distill_warm_step = distill.distill_warm_step
        self.debug = distill.get("debug",False)

        # self.distill_fpn_type = distill.get('distill_fpn_type', None)
        # self.distill_logits_type = distill.get("distill_logits_type", None)
        self.distill_reg_type = distill.get("distill_reg_type", None)
        self.num_classes = 80
        self.feat_channels = 256
        self.num_anchor = 9
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        '''
        gt_bboxes: list[tensor] 2[num, 4]
        gt_labels: list[tensor] 2[num, 1]
        '''
        x = self.extract_feat(img)
        losses, cls_reg_targets = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, teacher=True)  # losses{'loss_cls', 'loss_bbox'}
           
        # --------------------------- recall label target ------------------------- #                     
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # labels_list: list[tensor] 5[batch_size, num_anchor*h*w] 5[2, 9*h*w] (K, A, 4)

        stu_feature_adap = self.stu_feature_adap(x)
        y = self.teacher.extract_feat(img)

        stu_bbox_outs = self.bbox_head(x)
        stu_cls_score = stu_bbox_outs[0]
        stu_reg_delta = stu_bbox_outs[1]  # reg

        tea_bbox_outs = self.teacher.bbox_head(y)
        tea_cls_score = tea_bbox_outs[0] # cls_score, bbox_pred
        tea_reg_delta = tea_bbox_outs[1] 
        # cls_score 5[2, 9*80, 100, 152]; bbox_pred 5[2, 9*4, 100, 152]

        layers = len(stu_cls_score)
        distill_feat_loss, distill_cls_loss = 0, 0

        for layer in range(layers):
            stu_cls_score_sigmoid = stu_cls_score[layer].sigmoid() # [2, 9*80, h, w]
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values # [2, h, w] TODO anchor-based方法 channel维度包含9anchor，取9*80 max的时候是否有问题
            mask = mask.detach()

            #----------------- foreground ------------------- #
            labels = labels_list[layer].flatten() # [2*9*h*w] anchor(K, A, 4)
            valid_idxs = labels >= 0
            num_valid = valid_idxs.sum()
            foreground_idxs = (labels >=0) & (labels != self.num_classes)
            num_foreground = foreground_idxs.sum()
            
            # for FPN
            _, _, h, w = stu_feature_adap[layer].shape
            # fpn_fore_idxs = labels_list[layer].reshape(2, h, w, 9)
            fpn_labels = labels_list[layer].reshape(-1, 9)  # todo: 地址连续
            fpn_idxs = (fpn_labels >=0) & (fpn_labels != self.num_classes)
            fpn_idxs = fpn_idxs.sum(dim=1)
            fpn_fore_idxs = fpn_idxs > 0  # 2*h*w
            # import pdb
            # pdb.set_trace()
            num_fpn_fore = fpn_fore_idxs.sum()
            # ------------------- end -------------------------- # 

            # -------------------------- feat_loss ---------------------- #
            if self.distill_fpn_type == 'nomask':
                feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
                distill_feat_loss += feat_loss.mean()
            elif self.distill_fpn_type == "frs":
                feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
                distill_feat_loss += (feat_loss * mask[:,None,:,:]).sum() / mask.sum()
            elif self.distill_fpn_type == 'foreground':
                # FPN + foreground
                pred_features = stu_feature_adap[layer]
                teacher_features = y[layer]
                pred_features = pred_features.permute(0, 2, 3, 1).reshape(-1, self.feat_channels) # [2, 256, h, w]-> [2*h*w, 256]
                teacher_features = teacher_features.permute(0, 2, 3, 1).reshape(-1, self.feat_channels)

                feat_loss = F.mse_loss(
                    pred_features[fpn_fore_idxs],
                    teacher_features[fpn_fore_idxs],
                    reduction="sum",
                    ) / max(1, num_fpn_fore * self.feat_channels)
                distill_feat_loss += feat_loss
            # ---------------------- end -------------------------------- # 

            # ------------------- cls_loss ------------------------------ #
            # logits + Option: valid / foreground / frs / None
            stu_cls_score_sigmoid = stu_cls_score_sigmoid.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # [2, 9*80, h, w] --> [2*h*w*num_anchor, 80]
            tea_cls_score_sigmoid = tea_cls_score_sigmoid.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            

            if self.distill_logits_type == 'valid':
                cls_loss = F.binary_cross_entropy(
                    stu_cls_score_sigmoid[valid_idxs], 
                    tea_cls_score_sigmoid[valid_idxs],
                    reduction='sum') / max(1, num_valid * self.num_classes)  # todo or tea_cls_score_sigmoid[valid_idxs].sum()
                distill_cls_loss += cls_loss
            elif self.distill_logits_type == 'foreground':
                cls_loss = F.binary_cross_entropy(
                    stu_cls_score_sigmoid[foreground_idxs], 
                    tea_cls_score_sigmoid[foreground_idxs],
                    reduction='sum') / max(1, num_foreground * self.num_classes)
                distill_cls_loss += cls_loss
            elif self.distill_logits_type == 'frs':
                mask = mask.flatten() # 2*h*w
                cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid, reduction='None')
                distill_cls_loss +=  (cls_loss * mask[:,None,:,:]).sum() / mask.sum()
            elif self.distill_logits_type == 'nomask':
                cls_loss = F.binary_cross_entropy(
                    stu_cls_score_sigmoid, 
                    tea_cls_score_sigmoid,
                    reduction='mean')
                distill_cls_loss += cls_loss
            else:  # None
                pass  
            
            # reg loss
            if self.distill_reg_type == 'nomask':
                pass
            elif self.distill_reg_type == 'frs': # 设计reg分支自己的mask
                pass
            elif self.distill_reg_type == 'foreground':
                pass
            else:  # None
                pass

        distill_feat_loss = distill_feat_loss * self.distill_feat_weight
        distill_cls_loss = distill_cls_loss * self.distill_cls_weight
        distill_reg_loss = distill_reg_loss * self.distill_reg_weight

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
        if self.distill_reg_weight:
            losses.update({"distill_reg_loss":distill_reg_loss})

        return losses
