import os
from .openclip_encoder import OpenCLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    """
    vision_tower_cfg: initialized together with pretrained llava
    - pretrained model config has attribute  'mm_vision_tower'
    """
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', None)
    ckpt_path = getattr(vision_tower_cfg, '_name_or_path', '.')
    vision_tower_path = os.path.join(ckpt_path, vision_tower)
    is_path_exists = os.path.exists(vision_tower_path)
    if is_path_exists:
        return OpenCLIPVisionTower(vision_tower_path, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
