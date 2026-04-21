from .abmil import ABMIL, MABMIL
from .naive import NaiveRegress

MODEL_REGISTRY = {
    "ABMIL": ABMIL,
    "MABMIL": MABMIL,
    "NaiveRegress": NaiveRegress,
}

BACKBONE_FEAT_DIM = {
        'GIGA': 1536,
        'UNI':1024,
        'CHIEF': 768,
        'LUNIT': 384,
        'CONCH':512,
        'HOPT':1536,
        'VIRCHOW': 2560,
        'ENLIGHT': 768,
        'PRISM': 1280,
        'MUSK': 1024,
        'PLIP':512,
        'BiomedCLIP': 512,
        'QUILT-B32': 512,
        'QUILT-B16': 512,
}
def build_visual_encoder(conf):
    model_name = conf.model
    if model_name == "MABMIL":        
        conf.D_feat = {key: BACKBONE_FEAT_DIM[key] for key in conf.backbone}
    else:
        conf.D_feat = BACKBONE_FEAT_DIM[conf.backbone]
        conf.D_inner = conf.D_feat//2

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown visual encoder: '{model_name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](conf)