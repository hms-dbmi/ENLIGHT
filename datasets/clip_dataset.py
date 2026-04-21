import os, glob
from collections import defaultdict
import pandas as pd
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from .clip_label_defs import LABEL_2_PROMPT


def load_spider_colon(database):
    base_dir = f'{database}/spider_colorectal/SPIDER-colorectal'
    image_dir = f'{base_dir}/images'
    label_2_prompt = LABEL_2_PROMPT['SPIDER_colon']
    labelf = f'{base_dir}/metadata.json'
    label = json.load(open(labelf, 'r'))
    label_imgname = defaultdict(list)
    for data in tqdm(label):
        label = data['class']
        label_imgname[label] += [data['image_name']]
    
    img_list = []
    gt_list = []
    for gtid, label in enumerate(label_2_prompt.keys()):
        imgs = label_imgname[label]
        imgs = [f'{image_dir}/{img}' for img in imgs if os.path.exists(f'{image_dir}/{img}')]
        print(f'{label}: {len(imgs)}')
        img_list += imgs
        gt_list += len(imgs)*[gtid]
    return img_list, gt_list, label_2_prompt


def load_aggc22(database):
    img_list, gt_list = [], []

    img_list = glob.glob(f'{database}/AGGC22_patch/test_336/*/*.png')
    for imgf in img_list:
        gt_list.append(int(imgf.split('/')[-2])-1) # 1,2,3,4 -> 0,1,2,3
    
    return img_list, gt_list, LABEL_2_PROMPT['AGGC22']

def load_sicapv2(database):
    img_dir = f'{database}/SICAPv2/images'
    label_path = f'{database}/SICAPv2/partition/Test/Test.xlsx'
    data = pd.read_excel(label_path)
    data = data.drop(columns=['G4C'])

    img_list, gtid_list = [], []

    for idx, row in data.iterrows():
        img_name = row['image_name']
        labels = row.values[1:]
        if np.sum(labels) == 1:
            gt = np.where(labels == 1)[0][0]
            gtid_list.append(gt)
            img_list.append(f'{img_dir}/{img_name}')

    labels = data.columns[1:].tolist()
    label_2_prompt = LABEL_2_PROMPT['SICAPv2']
    return img_list, gtid_list, label_2_prompt

def load_rcckmc(database):
    label_2_prompt = LABEL_2_PROMPT['RCC-KMC']
    labels = label_2_prompt.keys()
    gt_list = []
    img_list = []
    for gt, label in enumerate(labels):
        imgs = glob.glob(f'{database}/*/*{label}/*.tif') + glob.glob(f'{database}/*/*{label}/*.jpg')+ glob.glob(f'{database}/*/*/*{label}/*.tif') + glob.glob(f'{database}/*/*/*{label}/*.jpg')
        img_list += imgs
        gt_list += [gt]*len(imgs)
        print(label, len(imgs))
    return img_list, gt_list, label_2_prompt


class ImageLabelDataset(Dataset):
    def __init__(self, img_list, label_list, preprocess):
        self.img_list = img_list
        self.preprocess = preprocess
        self.label_list = label_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.label_list[idx]
        image = self.preprocess(image)
        return image, label, img_path
    
TEMPLATE= [
    "CLASSNAME is present.",
    "a histopathological photograph of CLASSNAME.",
    "an H&E stained image of CLASSNAME."
]

dataload_func = {
    'SPIDER_colon': load_spider_colon,
    'AGGC22': load_aggc22,
    'RCC-KMC': load_rcckmc,
    'SICAPv2': load_sicapv2,
}

def load_data_clip(database, dataname, img_processor, batch_size = 128, num_workers=16):
    img_list, gtid_list, label_2_prompt = dataload_func[dataname](database)
    dataset = ImageLabelDataset(img_list, gtid_list, img_processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f'****{dataname}:{len(dataset)}****')
    return dataloader, label_2_prompt


##### Retrieval ######
class ImageCaptionDataset(Dataset):
    def __init__(self, img_list, cap_list, capidx_list, preprocess):
        self.img_list = img_list
        self.preprocess = preprocess
        self.cap_list = cap_list
        self.capidx_list = capidx_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path).convert('RGB')
        captionidx = self.capidx_list[idx]
        image = self.preprocess(image)
        return image, captionidx
    

def load_ut_pairs(database, data):
    """
    - 0-> 0.5 μm/pixel
    - 1-> 0.6 μm/pixel
    - 2-> 0.7 μm/pixel
    - 3-> 0.8 μm/pixel
    - 4-> 0.9 μm/pixel
    - 5-> 1.0 μm/pixel
    """
    mpp_version = data.split('-')[-1]
    images = sorted(glob.glob(f'{database}/*/{mpp_version}/*/*.jpg'))
    captions = [p.split('/')[-4] for p in images]
    capsets = list(set(captions))
    print({cap:sum([c==cap for c in captions]) for cap in capsets})
    img2cap_idx = [capsets.index(c) for c in captions]
    return images, capsets, img2cap_idx

def load_data_retreival(img_processor, data, batch_size = 128, num_workers=32):
    imagefs, captions, img2capidx = load_ut_pairs(data)
    dataset = ImageCaptionDataset(imagefs, captions, img2capidx, preprocess=img_processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader