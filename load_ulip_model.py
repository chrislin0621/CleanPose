from model.ULIP_models import ULIP_PointBERT,ULIP_PN_SSG
from easydict import EasyDict
import yaml
import torch
from collections import OrderedDict


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

def load_ULIP_Pointbert():
    config_addr = 'model/pointbert/PointTransformer_8192point.yaml'
    ulip2_pointbert_weights_path = 'model/pointbert/pretrained_model/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt'
    config = cfg_from_yaml_file(config_addr)


    ULIP_model = ULIP_PointBERT(config=config).cuda()
    state_dict = torch.load(ulip2_pointbert_weights_path)["state_dict"]
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    # unfreeze_list = ["point_encoder.pos_embed2","point_encoder.blocks.adapter_mlp"]
    ULIP_model.load_state_dict(new_state_dict, strict=True)
    for param in ULIP_model.parameters():
        param.requires_grad = False
    
    return ULIP_model

def load_ULIP_PN():
    ulip2_pointnet_weights_path = 'model/pointnet2_ulip/pretrained_model/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointnet2_ssg.pt'
    ULIP_PN_model = ULIP_PN_SSG().cuda()
    state_dict = torch.load(ulip2_pointnet_weights_path)["state_dict"]
    # edit keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
        
    ULIP_PN_model.load_state_dict(new_state_dict, strict=True)
    for param in ULIP_PN_model.parameters():
        param.requires_grad = False
    
    return ULIP_PN_model


# if __name__ == "__main__":