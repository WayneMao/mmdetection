from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
from torch import nn
import torch.nn.functional as F
# from mmdet.apis.inference import init_detector

@DETECTORS.register_module()
class Distilling_FRS_Single_CABlock(SingleStageDetector):
    '''
    Update:
    FPN + CA Block
    no mask
    '''

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_FRS_Single_CABlock, self).__init__(backbone, neck, bbox_head, train_cfg,
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

        # init CA Block
        fpn_channel = 256
        self.ca_bloacks = CA_Blocks(channel=fpn_channel)
        # self.ca_bloacks.append(CA_Block(channel=fpn_channel, h=100, w=152))
        # self.ca_bloacks.append(CA_Block(channel=fpn_channel, h=50, w=76))
        # self.ca_bloacks.append(CA_Block(channel=fpn_channel, h=25, w=38))
        # self.ca_bloacks.append(CA_Block(channel=fpn_channel, h=13, w=19))
        # self.ca_bloacks.append(CA_Block(channel=fpn_channel, h=7, w=10))
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        # CA Block
        x = self.ca_bloacks(x)

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
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values
            mask = mask.detach()

            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            # cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='none')
            cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='mean')

            # distill_feat_loss += (feat_loss * mask[:,None,:,:]).sum() / mask.sum()
            # distill_cls_loss +=  (cls_loss * mask[:,None,:,:]).sum() / mask.sum()

            # no mask baseline 2x
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

class CA_Blocks(nn.Module):
    '''
    CA Block
    https://arxiv.org/pdf/2103.02907.pdf 
    '''
    def __init__(self, channel, num_layers=5, reduction=16):
        super(CA_Blocks, self).__init__()

        # self.h = [100, 50, 25, 13, 7]
        # self.w = [152, 76, 38, 19, 10]

        self.num_layers = num_layers
        self.avg_pool_x = nn.ModuleList()
        self.avg_pool_y = nn.ModuleList()
        self.conv_1x1 = nn.ModuleList()
        self.F_h = nn.ModuleList()
        self.F_w = nn.ModuleList()
        for i in range(num_layers):
            self.avg_pool_x.append(nn.AdaptiveAvgPool2d((None, 1)))  # (h,1) (self.h[i], 1)
            self.avg_pool_y.append(nn.AdaptiveAvgPool2d((1, None)))  # (1,w)

            self.conv_1x1.append(conv_1x1_bn(channel, channel//reduction))
            # self.conv_1x1.append(nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False))
            # self.relu = nn.ReLU()
            # self.bn = nn.BatchNorm2d(channel//reduction)

            self.F_h.append(nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False))
            self.F_w.append(nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False))

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, inputs):
        out = []
        for i in range(self.num_layers):
            n,c,h,w = inputs[i].size()
            x_h = self.avg_pool_x[i](inputs[i]).permute(0, 1, 3, 2)
            x_w = self.avg_pool_y[i](inputs[i])

            x_cat_conv_relu = self.conv_1x1[i](torch.cat((x_h, x_w), 3))

            x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

            s_h = self.sigmoid_h(self.F_h[i](x_cat_conv_split_h.permute(0, 1, 3, 2)))
            s_w = self.sigmoid_w(self.F_w[i](x_cat_conv_split_w))

            out.append(inputs[i] * s_h.expand_as(inputs[i]) * s_w.expand_as(inputs[i]))

        return out

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))  
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out