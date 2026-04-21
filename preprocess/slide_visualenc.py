import argparse
import os
from tqdm import tqdm
import h5py
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from utils.constants import CACHE_DIR
from datasets.slide_dataset import LoadTileH5Dataset
from models.visual_encoder.backbone import load_model_preprocess, forward_backbone, VISUAL_BACKBONE
from preprocess.slide_tile import crop_slide_mpp
    
def extract_slide_grid_feature(encoder, slide_patch_h5f, slide_patch_weights=None):
    dataset = LoadTileH5Dataset(path=slide_patch_h5f, preprocess=encoder.image_processor, is_hf_processor=False)
    dataloader = DataLoader(dataset,
                    batch_size=256 if encoder.device=='cuda' else 16,
                    num_workers=2,
                    shuffle=False,
                    drop_last=False)
    if slide_patch_weights is not None:
        assert len(dataset) == len(slide_patch_weights), f'Patch Number not aligned: {len(dataset)}, {len(slide_patch_weights)}'
        slide_patch_weights = torch.tensor(slide_patch_weights)
    feat_sum = None
    n_patches = 0
    offset = 0
    for image, _ in tqdm(dataloader):
        feat = encoder.encode_images(image).detach().cpu()  # (B, d)
        B = feat.shape[0]
        if slide_patch_weights is None:
            feat_sum = feat.sum(dim=0) if feat_sum is None else feat_sum + feat.sum(dim=0)
            n_patches += B
        else:
            w = slide_patch_weights[offset: offset + B].to(feat.dtype)  # (B,)
            weighted = (feat * w[:,None,None]).sum(dim=0)  # (d,)
            feat_sum = weighted if feat_sum is None else feat_sum + weighted
            n_patches += B
            offset += B

    if slide_patch_weights is None:
        avg_feat = feat_sum / n_patches  # (d,)
    else:
        avg_feat = feat_sum / slide_patch_weights.sum()  # (d,) weighted mean
    return avg_feat

def infer_slide_feats(backbone, slide_patch_path, feat_file, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, preprocess = load_model_preprocess(backbone=backbone,
                                              ckpt_path=args.ckpt_path,
                                              cache_dir=args.cache_dir)
    model = model.to(device).eval()

    dataset = LoadTileH5Dataset(path=slide_patch_path, preprocess=preprocess, is_hf_processor=False)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            drop_last=False)
    
    all_feats = []
    for imgs, coords in tqdm(dataloader):
        with torch.no_grad():
            feat = forward_backbone(backbone, model, imgs.to(device)).detach().cpu()
        all_feats.append(feat)

    all_feats = torch.cat(all_feats, dim=0).numpy()
    with h5py.File(feat_file, 'a') as f:
        if backbone in f:
            del f[backbone]
        f.create_dataset(backbone, data=all_feats)
        
def get_arguments():
    parser = argparse.ArgumentParser('visualenc_enlight', add_help=False)
    parser.add_argument("--slide-path", type=str, required=True)
    parser.add_argument("--slide-cropped", action="store_true", default=True)
    parser.add_argument("--backbone", type=str, choices=VISUAL_BACKBONE, required=True)
    parser.add_argument('--output-path', default=f'{CACHE_DIR}/feat', type=str)
    parser.add_argument('--cache-dir', default=CACHE_DIR, type=str)
    parser.add_argument('--ckpt-path', default="", type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument("--desired-mpp", type=float, default=0.5)
    parser.add_argument("--hf-token", type=str, default='')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Crop if whole slide
    os.makedirs(args.cache_dir, exist_ok=True)
    slide_stem = Path(args.slide_path).stem  
    if args.slide_cropped:
        slide_patch_path = args.slide_path
    else:
        slide_patch_path = Path(args.cache_dir).resolve() / f"{slide_stem}.h5"  
        if not os.path.exists(slide_patch_path):
            crop_slide_mpp(args.slide_path, slide_patch_path, desired_mpp=args.desired_mpp, cache_dir = args.cache_dir)
    
    # Feature output
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    print(f'Saving to {args.output_path}')
    infer_slide_feats(args.backbone, slide_patch_path, args.output_path, args)
    print(f'Saved to {args.output_path}')

