import argparse
from collections import OrderedDict

import torch


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def migrate_ckpt(state_dict: OrderedDict) -> OrderedDict:
    """Migrate orginal grounding dino checkpoint to the one that suits current codebase. All we
    need is to rename the `backbone.0` to `backbone.model.backbone.`

    Args:
        state_dict (OrderedDict): orginal checkpoint.

    Returns:
        OrderedDict: migrated checkpoint.
    """
    new_state_dict = OrderedDict()
    ema_model = state_dict["ema_model"]
    ema_model = clean_state_dict(ema_model)
    # rename backbone.0 tp backbone.model.backbone
    for k, v in ema_model.items():
        if "universal_token" in k:
            new_state_dict[k.replace("universal_token", "fine_grained_prompt")] = v
        elif "universal_object_token" in k:
            new_state_dict[
                k.replace("universal_object_token", "coarse_grained_prompt")
            ] = v
        elif "counting_deform_layer" in k:
            continue
        elif "logit_scale" in k:
            continue
        elif "label_enc" in k:
            continue
        elif "indicator" in k:
            continue
        elif "aggregator" in k:
            continue
        else:
            new_state_dict[k] = v

    return dict(model=new_state_dict)


def get_args():
    parser = argparse.ArgumentParser(
        "Migrate GroundingDINO checkpoint to current codebase"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/comp_robot/jiangqing/projects/2023/research/T-Rex3/work_dirs/universal_proposal_only/v6_sam_clean_data_with_mosaic_data_object_token_swinl_loadgdinockpt/checkpoints/checkpoint_s0000017499.pth",
        help="path to the orginal checkpoint",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="checkpoints/upn_large.pth",
        help="path to the output checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    ckpt = migrate_ckpt(ckpt)
    torch.save(ckpt, args.out_path)
    print(f"Checkpoint saved to {args.out_path}")
