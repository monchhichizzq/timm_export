import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
# from os.path import dirname, realpath, sep, pardir
# abs_path = dirname(realpath(__file__)) + sep + pardir
# sys.path.append(abs_path)
print(sys.path)
import torch
import json
import numpy as np
from custom_models.efficientnet_sp.efficientnet_sp import EfficientNet
from custom_models.efficientnet_sp._efficientnet_builder import round_channels
from utils import set_lite_threshold

#   'efficientnet-lite0': (1.0, 1.0, 224, 0.2), 160 / 224
#   'efficientnet-lite1': (1.0, 1.1, 240, 0.2), 172 / 224
#   'efficientnet-lite2': (1.1, 1.2, 260, 0.3), 192 / 256 / 260
#   'efficientnet-lite3': (1.2, 1.4, 280, 0.3), 210 / 280
#   'efficientnet-lite4': (1.4, 1.8, 300, 0.3), 224 / 288 / 300

############# set params #############
# val_img_size = 280
# arch_name = "/home/zzq/Documents/apps/image_cls/pytorch-image-models/prune/tf_efficientnet_lite3_0.64G_A3/20250504-113341-tf_efficientnet_lite3-210/best_arch.pth"
# model_path = "/home/zzq/Documents/apps/image_cls/pytorch-image-models/prune/tf_efficientnet_lite3_0.64G_A3/20250504-113341-tf_efficientnet_lite3-210/model_best.pth.tar"
# cfg_path = "/home/zzq/Documents/apps/image_cls/pytorch-image-models/tf_export/sp_model_cfg/lite3_0.64G_A3.json"
# model_name = "lite3_0.64G_A3"

val_img_size = 280
arch_name = "/home/zzq/Documents/apps/image_cls/pytorch-image-models/prune/tf_efficientnet_lite3_0.9G_A3/20250504-113152-tf_efficientnet_lite3-210/best_arch.pth"
model_path = "/home/zzq/Documents/apps/image_cls/pytorch-image-models/prune/tf_efficientnet_lite3_0.9G_A3/20250504-113152-tf_efficientnet_lite3-210/model_best.pth.tar"
cfg_path = "/home/zzq/Documents/apps/image_cls/pytorch-image-models/tf_export/sp_model_cfg/lite3_0.9G_A3.json"
model_name = "lite3_0.9G_A3"

print("model_name:", model_name)
log_dir = "sp_onnx"
os.makedirs(log_dir, exist_ok=True)

############# build new model #############
print(f"[Stage-1] Building new model")
with open(cfg_path, 'r') as f:
    model_cfg = json.load(f)

def no_round_channels(x, **kwargs):
    return x

sp_model = EfficientNet(
    block_args = model_cfg["stages"],
    stem_size = model_cfg["stem_size"],
    num_features = model_cfg["num_features"],
    round_chs_fn = no_round_channels,
)
from torchsummary import summary
summary(sp_model, (3, val_img_size, val_img_size))


############# load sp weights #############
print(f"[Stage-5] Loading model weights from {model_path}")
checkpoint = torch.load(model_path, map_location='cpu')
print(checkpoint.keys())
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    sp_model.load_state_dict(state_dict, strict=True)
    print("loading from state_dict with strict=True")
else:
    sp_model.load_state_dict(checkpoint, strict=True)
    print("loading from checkpoint with strict=True")

torch_model_dict = set_lite_threshold(sp_model)

############# load sp weights #############
sp_model.eval()
dummy_input = torch.randn(1, 3, val_img_size, val_img_size)

onnx_file_path = os.path.join(log_dir, f"{model_name}.onnx")

torch.onnx.export(
    sp_model,                      # Model to be exported
    dummy_input,                # Dummy input for tracing
    onnx_file_path,             # Path to save the ONNX model
    export_params=True,         # Store the trained parameter weights inside the model file
    opset_version=10,           # ONNX version to export the model (default is 10)
    do_constant_folding=True,   # Optimize constants during export
    input_names=['input'],      # Input tensor name (optional)
    output_names=['output'],    # Output tensor name (optional)
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Optional for dynamic batch size
)
print(f"Model successfully exported to {onnx_file_path}")

############# check the onnx model #############
import onnx
import onnxruntime as ort
onnx_model = onnx.load(onnx_file_path)
onnx.checker.check_model(onnx_model)

# Error check
# step 1: prepare the input data
input_data = np.random.randn(1, 3, val_img_size, val_img_size).astype(np.float32)

# step 2: run torch inference
# convert numpy array to torch tensor
torch_input = torch.from_numpy(input_data)
# run inference
torch_output = sp_model(torch_input)
print(torch_output.shape)
# step 3: load the onnx model
session = ort.InferenceSession(onnx_file_path)
# You would need to preprocess the input data to match the model's input format
input_name = session.get_inputs()[0].name
# step 4: run inference
onnx_output = session.run(None, {input_name: input_data})
print("Onnx output length: ", len(onnx_output))

from onnx2keras import onnx_to_keras
# convert to keras
keras_model = onnx_to_keras(
    onnx_model, ['input'],
    change_ordering=True, 
    name_policy='keep',
    verbose=True
)
keras_model.summary()

# save keras model as .h5
keras_file_path = os.path.join(log_dir, f"{model_name}.h5")
keras_model.save(keras_file_path)
print(f"Model successfully exported to {keras_file_path}")

import tensorflow as tf
model = tf.keras.models.load_model(keras_file_path)
print("Model successfully loaded from", keras_file_path)

# save configs
outputs = ["output"]

model_info = {}
layer_index = 0
torch_layers = list(torch_model_dict.keys())
                            
for layer in keras_model.layers:
    layer_cls = layer.__class__.__name__
    if layer_cls in ["Conv2D", "DepthwiseConv2D"]:
        layer_name = layer.name
        torch_name = torch_layers[layer_index]
        output_shape = layer.output_shape
        h = output_shape[1]

        model_info[layer_name] = {}
        model_info[layer_name]["threshold"] = torch_model_dict[torch_name]["threshold"]
        model_info[layer_name]["torch_name"] = torch_name
        model_info[layer_name]["layer_type"] = "stateless"
        model_info[layer_name]["keras_type"] = layer_cls
        model_info[layer_name]["output_shape"] = output_shape[1:]

        # skip threshold outputs
        if layer_name in outputs:
            model_info[layer_name]["threshold"] = -126
            model_info[layer_name]["torch_name"] = torch_name
            model_info[layer_name]["layer_type"] = "stateless"
            model_info[layer_name]["keras_type"] = layer_cls
            model_info[layer_name]["output_shape"] = output_shape[1:]
        
        layer_index += 1
    else:
        print("skip:", layer_cls)

assert len(torch_layers) == layer_index

config_file_path = os.path.join(log_dir, f"{model_name}.json")

with open(config_file_path, 'w') as f:
    json.dump(model_info, f, indent=4)
print(f"Threshold config successfully exported to {config_file_path}")





# model = EfficientNet(
#     block_args=lite3_blc_cfg,
#     stem_size = 32
# )
# from torchsummary import summary
# summary(model, (3, 280, 280))

# model = tf_efficientnet_lite3(pretrained=False, blc_cfg=None)