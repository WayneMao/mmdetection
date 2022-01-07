import torch  # 命令行是逐行立即执行的
teacher_model_path20 = '/data/GitHub/mmdetection/minimodel/retinanet_r101_fpn_2x_20coco/epoch_24.pth'
teacher_model_path='/data/GitHub/mmdetection/model/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth'
content = torch.load(teacher_model_path)
print(content.keys())   # dict_keys(['meta', 'state_dict'])
print(content['meta'].keys())

# cpu error
# content1 = torch.load(teacher_model_path20)  # error cpu
# # dict_keys(['meta', 'state_dict', 'optimizer'])
# print(content1.keys())

device = torch.device('cpu')
content1 = torch.load(teacher_model_path20, map_location=device)
print(content1.keys())
print(content1['meta'].keys())