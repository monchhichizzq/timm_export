import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
# from os.path import dirname, realpath, sep, pardir
# abs_path = dirname(realpath(__file__)) + sep + pardir
# sys.path.append(abs_path)
print(sys.path)
import torch
import numpy as np
from custom_models_save.efficientnet_sp.efficientnet_sp import EfficientNet
from timm.models._efficientnet_blocks import InvertedResidual
from timm.models._efficientnet_blocks import DepthwiseSeparableConv
from sp_model_cfg.example_blc_cfg import lite4_blc_cfg as lite3_blc_cfg

#   'efficientnet-lite0': (1.0, 1.0, 224, 0.2), 160 / 224
#   'efficientnet-lite1': (1.0, 1.1, 240, 0.2), 172 / 224
#   'efficientnet-lite2': (1.1, 1.2, 260, 0.3), 192 / 256 / 260
#   'efficientnet-lite3': (1.2, 1.4, 280, 0.3), 210 / 280
#   'efficientnet-lite4': (1.4, 1.8, 300, 0.3), 224 / 288 / 300

############# set params #############
# model_name = "lite3_0.64G_A3"
# val_img_size = 280
# arch_name = "pytorch-image-models/prune/tf_efficientnet_lite3_0.64G_A3/20250504-113341-tf_efficientnet_lite3-210/best_arch.pth"
# model_path = "pytorch-image-models/prune/tf_efficientnet_lite3_0.64G_A3/20250504-113341-tf_efficientnet_lite3-210/model_best.pth.tar"

# model_name = "lite3_0.9G_A3"
# val_img_size = 280
# arch_name = "pytorch-image-models/prune/tf_efficientnet_lite3_0.9G_A3/20250504-113152-tf_efficientnet_lite3-210/best_arch.pth"
# model_path = "pytorch-image-models/prune/tf_efficientnet_lite3_0.9G_A3/20250504-113152-tf_efficientnet_lite3-210/model_best.pth.tar"

model_name = "lite4_1.24G_A3"
val_img_size = 280
arch_name = "pytorch-image-models/prune/tf_efficientnet_lite3_0.9G_A3/20250504-113152-tf_efficientnet_lite3-210/best_arch.pth"
model_path = "pytorch-image-models/prune/tf_efficientnet_lite3_0.9G_A3/20250504-113152-tf_efficientnet_lite3-210/model_best.pth.tar"


cfg_dir = "sp_model_cfg"
os.makedirs(cfg_dir, exist_ok=True)


############# load sp model #############
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

print(f"[Stage-1] Loading pruned model from {arch_name}")
model = torch.load(arch_name, map_location='cpu', weights_only=False) 

from torchsummary import summary
summary(model, (3, val_img_size, val_img_size))

############# build new sp block list #############
print(f"[Stage-2] Building sp block list from {arch_name}")
ds_block_dict = {
    'block_type': 'ds', 
    'out_chs': 16, 
    'stride': 1, 
    'act_layer': None, 
    'dw_kernel_size': 3, 
    'pw_kernel_size': 1, 
    'se_ratio': 0.0, 
    'pw_act': False, 
    'noskip': False, 
    's2d': False
}
ir_block_dict = {
    'block_type': 'ir', 
    'out_chs': 24, 
    'stride': 2, 
    'act_layer': None, 
    'dw_kernel_size': 3, 
    'exp_kernel_size': 1, 
    'pw_kernel_size': 1, 
    'exp_ratio': 6.0, 
    'se_ratio': 0.0, 
    'noskip': False, 
    's2d': False
}

sp_blocks = []
for name, module in model.named_modules():
 
    if "conv_stem" in name:
        stem_out = module.out_channels
    if "conv_head" in name:
        head_out = module.out_channels

    if isinstance(module, DepthwiseSeparableConv):
        block_dict = {}
        block_dict["block_type"] = "ds"
        block_dict["mid_chs"] = 0
        block_dict["out_chs"] = module.conv_pw.out_channels
        block_dict["stride"] = module.conv_dw.stride[0]
        block_dict["act_layer"] = None
        block_dict["dw_kernel_size"] = module.conv_dw.kernel_size[0]
        block_dict["pw_kernel_size"] = module.conv_pw.kernel_size[0]
        block_dict["se_ratio"] = 0.0
        block_dict["pw_act"] = False
        block_dict["noskip"] = False
        block_dict["s2d"] = False
        sp_blocks.append(block_dict)

    if isinstance(module, InvertedResidual):
        in_chs = module.conv_pw.in_channels
        mid_chs = module.conv_pw.out_channels
        out_chs = module.conv_pwl.out_channels
        exp_ratio = mid_chs / in_chs

        block_dict = {}
        block_dict["block_type"] = "ir"
        block_dict["mid_chs"] = mid_chs
        block_dict["out_chs"] = out_chs
        block_dict["stride"] = module.conv_dw.stride[0]
        block_dict["act_layer"] = None
        block_dict["dw_kernel_size"] = module.conv_dw.kernel_size[0]
        block_dict["exp_kernel_size"] = 1
        block_dict["pw_kernel_size"] = 1
        block_dict["exp_ratio"] = exp_ratio
        block_dict["se_ratio"] = 0.0
        block_dict["noskip"] = False
        block_dict["s2d"] = False
        sp_blocks.append(block_dict)
        print(block_dict)

print(sp_blocks)

############# reconstruct new sp block list with stage #############
print(f"[Stage-3] Reconstructing sp block list with stages")
old_blocks = []
sp_lite3_blc_cfg = []
block_index = 0
for stage in lite3_blc_cfg:
    stage_blocks = []
    for block in stage:
        old_blocks.append(block)
        stage_blocks.append(sp_blocks[block_index])
        block_index += 1
    
    sp_lite3_blc_cfg.append(stage_blocks)
    
assert len(sp_blocks) == len(old_blocks)
print(len(sp_blocks), len(old_blocks))

# print(sp_lite3_blc_cfg)

############# save a json #############
import json
json_path = os.path.join(cfg_dir, f"{model_name}.json")
print(f"[Stage-6] Saving model cfg into {json_path}")
# Save to JSON file
sp_dict = {}
sp_dict["stem_size"] = stem_out
sp_dict["num_features"] = head_out
sp_dict["stages"] = sp_lite3_blc_cfg
print(sp_dict) 

with open(json_path, 'w') as f:
    json.dump(sp_dict, f)


############# build new model #############
print(f"[Stage-4] Building new model")
sp_model = EfficientNet(
    block_args = sp_lite3_blc_cfg,
    stem_size = stem_out,
    num_features = head_out
)
from torchsummary import summary
summary(sp_model, (3, val_img_size, val_img_size))


############# load sp weights #############
print(f"[Stage-5] Loading model weights from {model_path}")
print(checkpoint.keys())
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    sp_model.load_state_dict(state_dict, strict=True)
    print("loading from state_dict with strict=True")
else:
    sp_model.load_state_dict(checkpoint, strict=True)
    print("loading from checkpoint with strict=True")







# model = EfficientNet(
#     block_args=lite3_blc_cfg,
#     stem_size = 32
# )
# from torchsummary import summary
# summary(model, (3, 280, 280))

# model = tf_efficientnet_lite3(pretrained=False, blc_cfg=None)