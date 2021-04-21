from mmseg.models import HRNet, ResNet, ResNetV1c 
import torch

# self=ResNet(depth=50) 
# self = ResNet(depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         dilations=(1, 1, 2, 4),
#         strides=(1, 2, 1, 1),
#         # norm_cfg=norm_cfg,
#         norm_eval=False,
#         style='pytorch',
#         contract_dilation=True)
self = ResNetV1c(depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        # norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True)

self.eval()
inputs = torch.rand(1,3,512,512)
level_outputs = self.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))
# (1, 64, 128, 128)
# (1, 128, 64, 64)
# (1, 256, 32, 32)
# (1, 512, 16, 16)
    

# extra = dict(
#     stage1=dict(
#         num_modules=1,
#         num_branches=1,
#         block='BOTTLENECK',
#         num_blocks=(4, ),
#         num_channels=(64, )),
#     stage2=dict(
#         num_modules=1,
#         num_branches=2,
#         block='BASIC',
#         num_blocks=(4, 4),
#         num_channels=(32, 64)),
#     stage3=dict(
#         num_modules=4,
#         num_branches=3,
#         block='BASIC',
#         num_blocks=(4, 4, 4),
#         num_channels=(32, 64, 128)),
#     stage4=dict(
#         num_modules=3,
#         num_branches=4,
#         block='BASIC',
#         num_blocks=(4, 4, 4, 4),
#         num_channels=(32, 64, 128, 256)))
# self = HRNet(extra, in_channels=1)
# self.eval()
# inputs = torch.rand(1, 1, 32, 32)
# level_outputs = self.forward(inputs)
# for level_out in level_outputs:
#     print(tuple(level_out.shape))

# #(1, 32, 8, 8)
# #(1, 64, 4, 4)
# #(1, 128, 2, 2)
# #(1, 256, 1, 1)




