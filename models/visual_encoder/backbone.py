import open_clip
from transformers import CLIPProcessor, CLIPModel
import os

VISUAL_BACKBONE = ['ENLIGHT', 'PLIP', 'BiomedCLIP',  'QUILT-B32', 'QUILT-B16', 'VIRCHOW',  'HOPT',  'MUSK', 'PRISM', 'CONCH', 'LUNIT', 'UNI', 'GIGA', 'CHIEF']

# ============================================================
# ENLIGHT
# ============================================================
def load_ENLIGHT_enc(ckpt_path, cache_dir, return_tokenizer=False):
    os.makedirs(cache_dir, exist_ok=True)

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14-336',
        pretrained=ckpt_path,
        cache_dir=cache_dir,
        force_quick_gelu=True,
    )
    model.visual.output_tokens = True
    if return_tokenizer:
        tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
        return model, preprocess, tokenizer
    else:
        return model, preprocess

def _forward_ENLIGHT(model, imgs):
    output = model.visual(imgs)
    return output[0]

# ============================================================
# PLIP
# ============================================================
def _load_PLIP(ckpt_path, cache_dir):
    model = CLIPModel.from_pretrained("vinid/plip", cache_dir=cache_dir)
    preprocessing = CLIPProcessor.from_pretrained("vinid/plip", cache_dir=cache_dir)
    # img_processor = partial(preprocessing.image_processor, return_tensors='pt')
    def img_processor(imgs):
        return preprocessing.image_processor(images=imgs, return_tensors='pt').pixel_values[0]
    return model, img_processor

def _forward_PLIP(model, imgs):
    return model.get_image_features(imgs)


# ============================================================
# BiomedCLIP / quiltb32 / quiltb16
# ============================================================
def _load_BiomedCLIP(ckpt_path, cache_dir):
    # from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, img_processor = open_clip.create_model_from_pretrained(model_name, cache_dir=cache_dir)
    return model, img_processor

def _load_QUILT_B32(ckpt_path, cache_dir):
    model,_, img_processor = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32', cache_dir=cache_dir)
    return model, img_processor

def _load_QUILT_B16(ckpt_path, cache_dir):
    model,_, img_processor = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-16', cache_dir=cache_dir)
    return model, img_processor

def _forward_clip_encode_image(model, imgs):
    return model.encode_image(imgs)


# ============================================================
# Registry  {name: (load_fn, forward_fn)}
# ============================================================
_REGISTRY = {
    # backbones — model loaded via load_clip_model; forward defined here
    'ENLIGHT':    (load_ENLIGHT_enc,  _forward_ENLIGHT),
    'PLIP':       (_load_PLIP, _forward_PLIP),
    'BiomedCLIP': (_load_BiomedCLIP, _forward_clip_encode_image),
    'QUILT-B32':   (_load_QUILT_B32,  _forward_clip_encode_image),
    'QUILT-B16':   (_load_QUILT_B16, _forward_clip_encode_image),
    # 'VIRCHOW':    (_load_VIRCHOW, _forward_VIRCHOW),
    # 'HOPT':       (_load_HOPT, _forward_generic),
    # 'GIGA':       (_load_GIGA, _forward_generic),
    # 'UNI':        (_load_UNI, _forward_generic),
    # 'CONCH':      (_load_CONCH, _forward_CONCH),
    # 'MUSK':       (load_MUSK, _forward_MUSK),
    # 'PRISM':      (load_PRISM, _forward_generic),
    # 'LUNIT':     (load_LUNIT, _forward_generic),
}

# ============================================================
# Public API
# ============================================================
def load_model_preprocess(backbone, ckpt_path, cache_dir):
    load_fn, _ = _REGISTRY[backbone]
    model, preprocess = load_fn(ckpt_path, cache_dir)
    model_size_mb = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 // (1024 ** 2)
    print(model_size_mb, 'MB')
    return model, preprocess


def forward_backbone(backbone, model, imgs):
    _, forward_fn = _REGISTRY[backbone]
    return forward_fn(model, imgs)