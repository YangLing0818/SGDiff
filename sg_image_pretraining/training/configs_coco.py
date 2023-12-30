import argparse
from dataclasses import dataclass
from typing import Tuple, Union, List, Optional

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_train_image_dir", type=str, default='/home/huangzl/Data/datasets/coco/images/train2017', help="Path to training dataset")
    parser.add_argument("--coco_val_image_dir", type=str, default='/home/huangzl/Data/datasets/coco/images/val2017', help="Path to training dataset")

    parser.add_argument("--coco_train_instances_json", type=str, default='/home/huangzl/Data/datasets/coco/annotations/instances_train2017.json', help="")
    parser.add_argument("--coco_train_stuff_json", type=str, default='/home/huangzl/Data/datasets/coco/annotations/stuff_train2017.json', help="")
    parser.add_argument("--coco_val_instances_json", type=str, default='/home/huangzl/Data/datasets/coco/annotations/instances_val2017.json', help="")
    parser.add_argument("--coco_val_stuff_json", type=str, default='/home/huangzl/Data/datasets/coco/annotations/stuff_val2017.json', help="")
    parser.add_argument("--coco_stuff_only", type=str, default='/home/huangzl/Data/datasets/vg/val.h5', help="")
    parser.add_argument("--image_size", type=int, default=224, help="Image size for training.")
    parser.add_argument("--mask_size", type=int, default=16, help="")
    parser.add_argument("--num_train_samples", type=int, default=None, help="")
    parser.add_argument("--num_val_samples", type=int, default=1024, help="")
    parser.add_argument("--include_relationships", type=bool, default=True, help="Use orphaned objects or not in the image.")
    parser.add_argument("--min_object_size", type=float, default=0.02, help="")
    parser.add_argument("--min_objects_per_image", type=int, default=3, help="")
    parser.add_argument("--max_objects_per_image", type=int, default=8, help="")
    parser.add_argument("--coco_include_other", type=bool, default=False, help="")
    parser.add_argument("--instance_whitelist", type=list, default=None, help="")
    parser.add_argument("--stuff_whitelist", type=list, default=None, help="")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--val_batch_size", type=int, default=128, help="Batch size per GPU for Validation.")

    parser.add_argument("--model_config_json", type=str, default='', help="Path to json file of model configs.")
    parser.add_argument("--graph_width", type=int, default=512, help="Width of Graph Tower.")
    parser.add_argument("--num_graph_layer", type=int, default=5, help="Number of layers in Graph Tower.")
    parser.add_argument("--embed_dim", type=int, default=512, help="Dimension of embeddings.")

    parser.add_argument("--name", type=str, default=None, help="Optional identifier for the experiment when storing logs. Otherwise use current time.")
    parser.add_argument("--workers", type=int, default=1, help="Number of dataloader workers per GPU.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=5.0e-4, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1.0e-8, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--warmup", type=int, default=10000, help="Number of steps to warmup for.")
    parser.add_argument("--use_bn_sync", default=False, action="store_true", help="Whether to use batch norm sync.")
    parser.add_argument("--skip_scheduler", action="store_true", default=False, help="Use this flag to skip the learning rate decay.")
    parser.add_argument("--save_frequency", type=int, default=1, help="How often to save checkpoints. epoch level.")
    parser.add_argument("--save_most_recent", action="store_true", default=False, help="Always save the most recent model trained to epoch_latest.pt.")
    parser.add_argument("--logs", type=str, default="./logs/", help="Where to store tensorboard logs. Use None to avoid storing logs.")
    parser.add_argument("--log_local", action="store_true", default=False, help="log files on local master, otherwise global master only.")

    parser.add_argument("--val_frequency", type=int, default=1, help="How often to run evaluation with val data.")
    parser.add_argument("--precision", choices=["amp", "amp_bfloat16", "fp16", "fp32"], default="amp", help="Floating point precision.")
    parser.add_argument("--pretrained", default='', type=str, help="Use a pretrained CLIP model weights with the specified tag or file path.")
    parser.add_argument("--pretrained-image", default=False, action='store_true', help="Load imagenet pretrained weights for image tower backbone if available.")

    parser.add_argument("--lock_image", default=False, action='store_true', help="Lock full image tower by disabling gradients.")
    parser.add_argument("--lock_image_unlocked_groups", type=int, default=0, help="Leave last n image tower layer groups unlocked.")
    parser.add_argument("--lock_image_freeze_bn_stats", default=False, action='store_true', help="Freeze BatchNorm running stats in image tower for any locked layers.")
    parser.add_argument('--image_mean', type=float, nargs='+', default=None, metavar='MEAN', help='Override default image mean value of dataset')
    parser.add_argument('--image_std', type=float, nargs='+', default=None, metavar='STD', help='Override default image std deviation of of dataset')
    parser.add_argument("--grad_checkpointing", default=False, action='store_true', help="Enable gradient checkpointing.")
    parser.add_argument("--local_loss", default=False, action="store_true", help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)")
    parser.add_argument("--gather_with_grad", default=False, action="store_true", help="enable full distributed gradient for feature gather")
    parser.add_argument("--force_quick_gelu", default=False, action='store_true', help="Force use of QuickGELU activation for non-OpenAI transformer models.")

    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--report_to", default='', type=str, help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']")
    parser.add_argument("--debug", default=False, action="store_true", help="If true, more information is logged.")
    parser.add_argument("--ddp_static_graph", default=False, action='store_true', help="Enable static graph optimization for DDP in PyTorch >= 1.11.")
    parser.add_argument("--no_set_device_rank", default=False, action="store_true", help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).")
    parser.add_argument("--seed", type=int, default=9768, help="Default random seed.")
    parser.add_argument("--norm_gradient_clip", type=float, default=None, help="Gradient clip.")
    args = parser.parse_args()

    return args

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int]
    width: int
    head_width: int
    image_size: int
    mlp_ratio: float
    patch_size: int = None
    timm_model_name: str = None
    timm_model_pretrained: bool = None
    timm_pool: str = None
    timm_proj: str = None


@dataclass
class CLIPGraphCfg:
    layers: int
    width: int