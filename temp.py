# from mmdet.apis import init_detector, inference_detector

# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# # 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# # 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# device = 'cuda:0'
# # 初始化检测器
# model = init_detector(config_file, checkpoint_file, device=device)
# # 推理演示图像
# inference_detector(model, 'demo/demo.jpg')


from mmdet.apis import init_detector, inference_detector
import mmcv

# # Specify the path to model config and checkpoint file
# config_file = 'configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py'
# checkpoint_file = 'checkpoints/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth'
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo/demo.jpg'
# img = mmcv.imread(img) # which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
# 可视化推理检测的结果
model.show_result(img, result)
# or save the visualization results to image files
# 将推理的结果保存
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
# 测试视频片段的推理结果
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)
