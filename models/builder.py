"""
Code adapated from https://github.com/haotian-liu/LLaVA/blob/main/llava/model/builder.py
Thanks to the authors of LLaVA
"""
import os
import warnings
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
import open_clip

from models import *
from utils.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

"""
 tokenizer, model, _, _ = load_pretrained_model(
        model_path, 
        args.model_base, 
        model_name,
        cache_dir=args.output_dir)
    model.config.tokenizer_padding_side = "left" 
"""

def load_enlight_model(
    lmm_path: str,
    cache_dir: str,
    device_map: str = "auto",
    device: str = "cuda",
    use_flash_attn: bool = False,
    bf16: bool = False,
):
    """
    Returns:
        tokenizer, model, image_processor, context_len
    """
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        lmm_path,
        load_8bit=False,
        load_4bit=False,
        device_map=device_map,
        device=device,
        cache_dir = cache_dir,
        use_flash_attn=use_flash_attn,
    )

    if bf16:
        model = model.to(torch.bfloat16)

    return tokenizer, model, image_processor, context_len

def load_pretrained_model(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path, cache_dir=kwargs.get('cache_dir', None))
    config.cache_dir = kwargs.get('cache_dir', None)
    model = LlavaLlamaForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                config=config,
                **kwargs
            )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != 'auto':
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return tokenizer, model, image_processor, context_len
