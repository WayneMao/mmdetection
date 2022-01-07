from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
# from mmdet.apis.inference import init_detector

@DETECTORS.register_module()
class Distilling_Single_Attention(SingleStageDetector):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_Single_Attention, self).__init__(backbone, neck, bbox_head, train_cfg,
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
        # attention
        # in_dim = 9*80
        # self.attention = SelfAttention(in_dim)
    
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
        
        # ------------- self attention ---------------- #
        '''两种思路： 点之间的关系
        1. 先对cls层面做self attention, 然后frs思路max取mask;(尽可能多的获取信息)
        2. frs 的mask层面直接做selfattention;
        3. mask层面直接QKV
        这里是3th
        '''
        # ----------------------- end ------------------ #

        for layer in range(layers):
            stu_cls_score_sigmoid = stu_cls_score[layer].sigmoid()
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()  # [n,9*80, h, w]
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values
            mask = self.mask_attention(mask[:,None,:,:])  # [n,1,h,w]
            mask = mask.detach()

            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='none')

            distill_feat_loss += (feat_loss * mask).sum() / mask.sum()  # mask[:,None,:,:]-> mask
            distill_cls_loss +=  (cls_loss * mask).sum() / mask.sum()

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

    def mask_attention(self, mask):
        # [n,1,h,w]
        query, key, value = mask, mask, mask

        m_batchsize, C, height, width = mask.size()
        
        query = query.view(m_batchsize, -1, width * height)  # [B, 1, H*W]
        key = key.view(m_batchsize, -1, width * height)  # [B, 1, H*W]
        value = value.view(m_batchsize, -1, width * height)  # [B, 1, H*W]
        
        attention = torch.bmm(query.permute(0, 2, 1), key)  # []^T--> [B, h*w, 1] * [B, 1, H*W] = [B, H*W, H*W]
        attention = attention.softmax(dim=-1)
        
        self_attetion = torch.bmm(value, attention)  # [B, 1, H*W] * [B, H*W, H*W] = [B, 1, H*W]
        self_attetion = self_attetion.view(m_batchsize, C, height, width)  # [B, 1, H, W]
        
        mask_out = self_attetion + mask  # todo delete: +x
        return mask_out


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class SelfAttention(nn.Module):
    r"""
        Self attention Layer.
        Source paper: https://arxiv.org/abs/1805.08318
    """
    def __init__(self, in_dim, activation=F.relu):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        # Q,K,V
        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)
        
    def forward(self, x):
        """
            inputs :
                x : input feature maps [B, C, H, W]
            returns :
                out : self attention feature maps
                
        """
        m_batchsize, C, height, width = x.size()
        
        f = self.f(x).view(m_batchsize, -1, width * height)  # [B, C//8, H*W]
        g = self.g(x).view(m_batchsize, -1, width * height)  # [B, C//8, H*W]
        h = self.h(x).view(m_batchsize, -1, width * height)  # [B, C, H*W]
        
        attention = torch.bmm(f.permute(0, 2, 1), g)  # []^T--> [B, h*w, C//8,] * [B, C//8, H*W] = [B, H*W, H*W]
        attention = self.softmax(attention)
        
        self_attetion = torch.bmm(h, attention)  # [B, C, H*W] * [B, H*W, H*W] = [B, C, H*W]
        self_attetion = self_attetion.view(m_batchsize, C, height, width)  # [B, C, H, W]
        
        out = self.gamma * self_attetion + x  # todo delete: +x
        return out