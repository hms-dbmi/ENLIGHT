import argparse
import os
from pathlib import Path 
import torch

from models.builder import load_enlight_model
from models.multimodal_encoder.openclip_encoder import OpenCLIPEnc
from preprocess.slide_tile import crop_slide_mpp
from preprocess.slide_visualenc import extract_slide_grid_feature
from datasets.lmm_dataset import format_conversation
from utils.mm_utils import tokenizer_image_token
from utils.constants import CACHE_DIR, IMAGE_TOKEN_INDEX

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def infer_model(slide_patch_path, args):
    # Load model
    disable_torch_init()
    tokenizer, model, _, _ = load_enlight_model(args.model_gen, cache_dir = args.cache_dir)
    model.config.tokenizer_padding_side = "left" 
    if args.bf16:
        model=model.to(torch.bfloat16)
    dtype = next(model.get_model().mm_projector.parameters()).dtype

    # Extract slide feature
    encoder = OpenCLIPEnc(model = model.get_vision_tower().clip,
                          image_processor = model.get_vision_tower().image_processor,
                          imgfeat_type='patch')
    img_feats = extract_slide_grid_feature(encoder, slide_patch_h5f=slide_patch_path)

    # Prepare text input
    prompt = format_conversation(args.question, 
                                args.use_image, 
                                conv_mode = args.conv_mode)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids[None, :].to(device='cuda', non_blocking=True)
    img_feats = img_feats[None, :].to(dtype=dtype, device='cuda', non_blocking=True)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=None,
            images=None,
            image_sizes=None,
            img_feats=img_feats,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True) 

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(outputs)


def get_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model-gen", type=str, default='./ckpts/enlight-fm')

    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    # Data
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--use-image", type=int, default=1)
    parser.add_argument("--slide-path", type=str, required=True)
    parser.add_argument("--slide-cropped", action="store_true", default=False,
                        help='True if cropped and saved in H5, otherwise False')

    parser.add_argument("--question", type=str, default="Summarize what you see.")
    parser.add_argument("--desired-mpp", type=float, default=0.5)

    # Setting
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    slide_stem = Path(args.slide_path).stem  
    if args.slide_cropped:
        slide_patch_path = args.slide_path
    else:                                                                                       
        slide_patch_path = Path(args.cache_dir).resolve() / f"{slide_stem}.h5"    
        if not os.path.exists(slide_patch_path):
            crop_slide_mpp(args.slide_path, slide_patch_path, desired_mpp=args.desired_mpp, cache_dir=args.cache_dir)

    infer_model(slide_patch_path, args)
