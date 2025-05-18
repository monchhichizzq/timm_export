


def replace_name(pretrained_dict):
    new_pretrained_dict = {}
    for name, param in pretrained_dict.items():
        if ".layer." in name:
            name = name.replace(".layer.", ".")
            new_pretrained_dict[name] = param
        else:
            new_pretrained_dict[name] = param
    return new_pretrained_dict


def set_lite_threshold(model):
    model_info = {}

    for name, module in model.named_modules():
        if "conv_stem" in name:
            model_info[name] ={}
            model_info[name]["threshold"] = -126
            model_info[name]["layer_type"] = "stateless"
        elif "conv_dw" in name:
            model_info[name] ={}
            if "blocks.0.0.conv_dw" in name:
                model_info[name]["threshold"] = -126 # -1
            else:  
                model_info[name]["threshold"] = -126 # -3
            model_info[name]["layer_type"] = "stateless"
        elif "conv_pwl" in name:
            model_info[name] ={}
            model_info[name]["threshold"] = -126 
            model_info[name]["layer_type"] = "stateless"    
        elif "conv_pw" in name:
            model_info[name] ={}
            model_info[name]["threshold"] = -126 # -3
            model_info[name]["layer_type"] = "stateless"
        elif "expand_pw.conv" in name:
            model_info[name] ={}
            model_info[name]["threshold"] = -126 # -1
            model_info[name]["layer_type"] = "stateless"
        elif "conv_head" in name:
            model_info[name] ={}
            model_info[name]["threshold"] = -126
            model_info[name]["layer_type"] = "stateless"
        elif "head_thr.conv" in name:
            model_info[name] ={}
            model_info[name]["threshold"] = -126 # -1
            model_info[name]["layer_type"] = "stateless"
        else:
            pass
    
    print(model_info.keys())
    return model_info