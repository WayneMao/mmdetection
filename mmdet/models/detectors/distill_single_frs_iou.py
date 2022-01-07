from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
from mmdet.core import build_bbox_coder, build_anchor_generator, images_to_levels
import torch
import torch.nn.functional as F
# from mmdet.apis.inference import init_detector

@DETECTORS.register_module()
class Distilling_Single_FRS_IoU(SingleStageDetector):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_Single_FRS_IoU, self).__init__(backbone, neck, bbox_head, train_cfg,
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

        # decoder
        '''
        1. 回归值delta 解码 --> bbox
        1.1 general anchor
        1.2 reshape
        2. bbox gt 计算IoU
        3. 用2得到的IoU 校正cls 分数来得到mask
        '''
        # anchor_generator=dict(
        #     type='AnchorGenerator',
        #     octave_base_scale=4,
        #     scales_per_octave=3,
        #     ratios=[0.5, 1.0, 2.0],
        #     strides=[8, 16, 32, 64, 128])
        # bbox_coder=dict(
        #     type='DeltaXYWHBBoxCoder',
        #     target_means=[.0, .0, .0, .0],
        #     target_stds=[1.0, 1.0, 1.0, 1.0])

        # self.bbox_coder = build_bbox_coder(bbox_coder)
        # self.anchor_generator = build_anchor_generator(anchor_generator)


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        x = self.extract_feat(img)  # backbone (+neck)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        stu_feature_adap = self.stu_feature_adap(x)
        y = self.teacher.extract_feat(img)

        stu_bbox_outs = self.bbox_head(x)
        stu_cls_score = stu_bbox_outs[0]

        tea_bbox_outs = self.teacher.bbox_head(y)  # cls + reg
        tea_cls_score = tea_bbox_outs[0]
        
        layers = len(stu_cls_score)
        distill_feat_loss, distill_cls_loss = 0, 0

        # --------------------- v2 ------------- #
        # tea_losses = self.teacher.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore)

        # -------------------- decoder + IoU v1----------------------- #
        # tea_bbox_delta = tea_bbox_outs[1]  # list(tensor) l=5   levels: 5[n, 9*4, h,w]
        # featmap_sizes = [featmap.size()[-2:] for featmap in tea_cls_score]  # list[Tensor] 5[h,w]
        # num_imgs = len(img_metas)
        # device = stu_cls_score[0].device
        # multi_level_anchors = self.anchor_generator.grid_anchors(
        #     featmap_sizes, device)  # list[Tensor]] 5[h*w*9, 4]
        # anchor_list = [multi_level_anchors for _ in range(num_imgs)]  # list[list[Tensor]] 2[5[h*w*9, 4]]
        # # anchor number of multi levels
        # num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]  # [136800, 34200, 8550, 2223, 630]
        # # concat all level anchors to a single tensor
        # concat_anchor_list = []
        # for i in range(num_imgs):
        #     concat_anchor_list.append(torch.cat(anchor_list[i]))  # 2[num_anchor, 4]
        # all_anchor_list = images_to_levels(concat_anchor_list,
        #                                    num_level_anchors) # 5[2, num', 4]
        # all_anchor_list = [item.reshape(-1, 4) for item in all_anchor_list]  # 5[n*h*w*9, 4] todo view
        # bbox_pred = [item.permute(0, 2, 3, 1).reshape(-1, 4) for item in tea_bbox_delta]  # 5[n,h,w,9*4] --> 5[n*h*w*9, 4]

        # # decoder def delta2bbox
        # all_bbox_xyxy = []  # 5[N,4]
        # for layer in range(layers):
        #     bbox_xyxy = self.bbox_coder.decode(all_anchor_list[layer], bbox_pred[layer])  # (N,4)
        #     all_bbox_xyxy.append(bbox_xyxy)
        
        # # get teacher bbox_target
        # losses = self.teacher.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore)
        
        # loss is dict, it can incloud loss_cls, loss_reg, bbox_targets
        # gt_bboxes --> box_targets  TODO
        
        # iou: (N,4)-> [N,1] --> [n,9,h,w]

        # ------------------------ end --------------------- #

        for layer in range(layers):  # 5[n,9*80, h, w]
            stu_cls_score_sigmoid = stu_cls_score[layer].sigmoid()
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()  # [n,9*80, h, w]
            n, _, h, w = tea_cls_score_sigmoid.shape
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values  # [n, h, w]
            mask = mask.detach()
            # self.bbox_coder

            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='none')

            distill_feat_loss += (feat_loss * mask[:,None,:,:]).sum() / mask.sum()
            distill_cls_loss +=  (cls_loss * mask[:,None,:,:]).sum() / mask.sum()

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

    def iou_aware(self, bbox_pred, bbox_target):
        pass