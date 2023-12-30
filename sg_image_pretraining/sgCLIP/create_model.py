import torch
import json
from typing import Optional, Tuple
from sgCLIP.model import sgCLIP, convert_weights_to_fp16


def create_model(
        args,
        graph_vocab: dict,
        model_config_json: str,
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        force_quick_gelu: bool = False,
        pretrained_image: bool = False,
):
    if model_config_json != '':
        with open(model_config_json, 'r') as f:
            model_cfg = json.load(f)
    else:
        model_cfg = {
            "graph_cfg": {
                "layers": args.num_graph_layer,
                "width": args.graph_width,
            },
            "embed_dim": args.embed_dim,
        }

    if force_quick_gelu:
        model_cfg["quick_gelu"] = True

    if pretrained_image:
        if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
            model_cfg['vision_cfg']['timm_model_pretrained'] = True
        else:
            assert False, 'pretrained image towers currently only supported for timm models'

    model = sgCLIP(graph_vocab=graph_vocab, **model_cfg)

    model.to(device=device)
    if precision == "fp16":
        assert device.type != 'cpu'
        convert_weights_to_fp16(model)

    return model

def create_model_and_transforms(
        args,
        graph_vocab: dict,
        model_config_json: str,
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        force_quick_gelu: bool = False,
        pretrained_image: bool = False,
):
    model = create_model(args, graph_vocab, model_config_json, precision, device, force_quick_gelu=force_quick_gelu, pretrained_image=pretrained_image)

    return model