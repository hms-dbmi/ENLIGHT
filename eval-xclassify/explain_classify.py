#!/usr/bin/env python
import h5py
import os
import yaml
from pprint import pprint
import numpy as np
import time  
import argparse
import torch
from pathlib import Path 

from models.visual_encoder import build_visual_encoder
from models.multimodal_encoder.openclip_encoder import OpenCLIPEnc
from models.builder import load_enlight_model
from preprocess.slide_visualenc import extract_slide_grid_feature
from preprocess.slide_tile import crop_slide_mpp
from datasets.lmm_dataset import format_conversation
from utils.mm_utils import tokenizer_image_token
from utils.constants import CACHE_DIR, IMAGE_TOKEN_INDEX


torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def infer_weights(net, device, slide_feats_dict):
    # Set the network to evaluation mode
    net.eval()
    slide_feats_dict = {k:torch.tensor(v).to(device, dtype=next(net.parameters()).dtype) for k,v in slide_feats_dict.items()}
    num_source = len(slide_feats_dict)
    num_tile = [v.shape[0] for v in slide_feats_dict.values()][0]

    # Predicted label
    wspred= net.infer_bag(slide_feats_dict)
    pred_label = wspred.argmax().item()

    # Attended weight
    weight = net.A.reshape(net.N_att, num_source, num_tile)
    weight = weight.mean(dim=0).sum(dim=0) #sum across feature sources
    weight = weight.detach().cpu().numpy()
            
    return pred_label, weight

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def setup_config(args):
    # init with yaml, overwrite by args
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    # Model
    conf.backbone = conf.backbone.split(',')
    conf.n_token = 1
    conf.n_masked_patch = 0
    conf.mask_drop = 0
    return conf

def get_arguments():
    parser = argparse.ArgumentParser('xclassify_enlight', add_help=False)
    parser.add_argument('--cache-dir', default=CACHE_DIR, type=str)
    parser.add_argument('--config', dest='config', default='eval-xclassify/config/GBM-LGG.yml',
                        help='settings of dataset in yaml format')
    parser.add_argument('--ckpt-dir', default="./ckpts", type=str)
    parser.add_argument("--desired-mpp", type=float, default=0.5)
    parser.add_argument('--slide-path', required=True, type=str)
    parser.add_argument("--slide-cropped", action="store_true", default=False,
                        help='True if cropped and saved in H5, otherwise False')
    parser.add_argument('--slide-multifeat-path', required=True, type=str)
    # for generative module
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # Load config file
    args = get_arguments()
    conf = setup_config(args)
    pprint(vars(conf))
    
    # Read Slide Feat
    with h5py.File(conf.slide_multifeat_path, 'r') as f:
        slide_feats = {key: np.array(f[key]) for key in f.keys()}
    for key, data in slide_feats.items():
        print(key, data.shape)
    
    # Extract slide feature for interpretation
    os.makedirs(conf.cache_dir, exist_ok=True)
    slide_stem = Path(conf.slide_path).stem    
    if conf.slide_cropped:
        slide_patch_path = conf.slide_path
    else:
        slide_patch_path = Path(conf.cache_dir).resolve() / f"{slide_stem}.h5"                                                                                  
        if not os.path.exists(slide_patch_path):
            crop_slide_mpp(conf.slide_path, slide_patch_path, desired_mpp=conf.desired_mpp, cache_dir=conf.cache_dir)

    # Load classification model
    model_cls = build_visual_encoder(conf).to(device)
    classifier_path = Path(conf.ckpt_dir) / conf.ckpt_classifier
    model_cls.load_state_dict(torch.load(classifier_path, map_location=torch.device(device)))
    
    # Load interpretation model
    lmm_path = Path(conf.ckpt_dir) / conf.ckpt_lmm
    tokenizer, model, _, _ = load_enlight_model(lmm_path, cache_dir = conf.cache_dir)
    model.config.tokenizer_padding_side = "left" 
    if conf.bf16:
        model=model.to(torch.bfloat16)
    dtype = next(model.get_model().mm_projector.parameters()).dtype
    slide_encoder = OpenCLIPEnc(model = model.get_vision_tower().clip,
                        image_processor = model.get_vision_tower().image_processor,
                        imgfeat_type='patch')

    # Cancer region identification
    if conf.cancer_prob_thresh>0:
        cancerous_slide_patch_path = Path(conf.cache_dir).resolve() / f"{slide_stem}-cancer{conf.cancer_prob_thresh}.h5"  
        slide_feats = slide_encoder.cancerous_patch_filter(slide_patch_path,
                                                    slide_all_feats = slide_feats, 
                                                    cancer_prob_thresh=conf.cancer_prob_thresh,
                                                    cancerous_slide_patch_path = cancerous_slide_patch_path)
    else:
        cancerous_slide_patch_path = slide_patch_path
    
    # Classification
    start = time.time()
    pred_label, slide_patch_weights = infer_weights(model_cls, device, slide_feats)

    classes = conf.classify_classes.split(',')
    pred_class = classes[pred_label]
    print(f"Finish classification: {(time.time()-start)/60:.2f} mins")

    # Interpret
    slide_feats = extract_slide_grid_feature(slide_encoder, 
                                        slide_patch_h5f = cancerous_slide_patch_path, 
                                        slide_patch_weights = slide_patch_weights)
    # Prepare text input
    question_prompt = 'Summarize what you see.'
    
    prompt = format_conversation(question_prompt, 
                                use_image = True, 
                                conv_mode = conf.conv_mode)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids[None, :].to(device='cuda', non_blocking=True)
    slide_feats = slide_feats[None, :].to(dtype=dtype, device='cuda', non_blocking=True)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=None,
            images=None,
            image_sizes=None,
            img_feats=slide_feats,
            do_sample=True if conf.temperature > 0 else False,
            temperature=conf.temperature,
            top_p=conf.top_p,
            num_beams=conf.num_beams,
            max_new_tokens=conf.max_new_tokens,
            use_cache=True) 

    generate_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f'Slide: {conf.slide_path}')
    print(f'Prediction: {pred_class}')
    print(f'Interpretation: {generate_text}')