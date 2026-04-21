# CONTROLLER_HEART_BEAT_EXPIRATION = 30
# WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Path
QA_DIR="./qafiles"
CACHE_DIR="./cache"

# Classify
CANCER_SUBTYPE_DEF = {
    'brca': 'Breast cancer.',
    'brcad':'Breast Ductal Carcinoma.',
    'brcal': 'Breast Lobular Carcinoma.',
    'lgg': 'Brain Lower Grade Glioma.',
    'gbm': 'Glioblastoma Multiforme.',
    'luad': 'Lung Adenocarcinoma.',
    'lusc': 'Lung Squamous Cell Carcinoma.',
    'kirp':'Renal Papillary Cell Carcinoma.',
    'kirc':'Renal Clear Cell Carcinoma.',
    'sov': 'Serious ovarian cancer.',
    'coad': 'Colon Adenocarcinoma.',
    'read': 'Rectum Adenocarcinoma'
}
CANCER_SUBTYPE_DATA = ['brcad-brcal', 'luad-lusc', 'kirp-kirc', 'gbm-lgg']

MUTATE_TYPES = {
    'amp': ['copy number diploid or deletion', 'copy number amplification'],
    'del': ['copy number diploid or amplification',  'copy number deletion'],
    'mutate': ['not mutated', 'mutated']
}