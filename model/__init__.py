from model.base import BevEncode
from .unet import Unet
from .unet_training import weights_init
import torch
import numpy as np

def get_model(method, inC, outC, instance_seg, embedding_dim,pretrained,backbone='vgg'):
    if method == 'BevEncode':
        model = BevEncode(inC, outC, instance_seg, embedding_dim, pretrained)
    
    elif method == 'unet':
        model_path="model_data"
        # 首先判断是否需要预训练参数
        if pretrained:
            from utils.unet_weight import download_weights
            download_weights(backbone,model_path)
        model = Unet(num_classes=outC, pretrained=pretrained, backbone=backbone)
        
        if not pretrained:
            weights_init(model)
        if model_path != '':
            print('Load weights {}.'.format(model_path))
            model_dict      = model.state_dict()
            pretrained_dict = torch.load(model_path)
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)

            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")


        
    else:
        raise NotImplementedError

    return model
