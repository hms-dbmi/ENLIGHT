import logging
import random
import os
import pandas
import numpy as np
from torch.utils.data import Dataset

from utils.io import read_hdf5
from utils import constants as cfg

_CANCER_SUBTYPE_DEF = cfg.CANCER_SUBTYPE_DEF
_CANCER_SUBTYPE_DATA =  cfg.CANCER_SUBTYPE_DATA
_MULTI_FEAT ={
    'all': ['giga', 'chief', 'uni', 'lunitp8'],
    'all5': ['giga', 'uni', 'hopt', 'lunitp8', 'virchow'],
    'all+ours0': ['giga', 'chief', 'uni', 'lunitp8', 'ours0'],
    'all+ours0+cp': ['giga', 'chief', 'uni', 'lunitp8', 'ours0', 'ours0_canprob'],
    'all7+ours0': ['giga', 'conch', 'chief', 'uni', 'lunitp8', 'virchow', 'hopt', 'ours0'],
    'all7+ours0+cp': ['giga', 'conch', 'chief', 'uni', 'lunitp8', 'virchow', 'hopt', 'ours0', 'ours0_canprob'],
    "a6+ours0": ['giga', 'conch', 'chief', 'uni', 'virchow', 'hopt', 'ours0'],
    "a5+ours0": ['giga', 'conch', 'uni', 'virchow', 'hopt','ours0'],
    "b6+ours0": ['giga', 'conch', 'chief', 'uni', 'virchow', 'lunitp8','ours0'],
    "b5+ours0": ['giga', 'conch', 'uni', 'virchow', 'lunitp8','ours0'],
}

_MUTATE_TYPES = cfg.MUTATE_TYPES

BASE_DIR = '/n/data2/hms/dbmi/kyu/lab/xug751'
DSMIL_CANCER_GENE_FILEDIR = {
    'mpp0.5': f'{BASE_DIR}/data/dsmil_cancer/gene_mpp0.5'}
DSMIL_FEAT_PATH = f'{BASE_DIR}/data/Feats'
DSMIL_CANCER_SUBTYPE_DIR = {
    'mpp0.5': f'{BASE_DIR}/data/dsmil_cancer/subtype_mpp0.5',}


def data_to_feat_dirs(project, data, feat_origin, data_base=''):
    assert project in ['tcga', 'dfciv2']
    breakpoint()

    if data_base:
        feat_path = DSMIL_FEAT_PATH.replace(BASE_DIR, data_base)
    else:
        feat_path = DSMIL_FEAT_PATH

    if data in _CANCER_SUBTYPE_DATA:
        data1, data2 = data.split('-')
        files = [f'{feat_path}/{feat_origin}/{project}_{data1}',
                f'{feat_path}/{feat_origin}/{project}_{data2}']
    else: # Gene task
        # cancer, mutemode, gene = data.split('-')
        cancer_mutemode, gene = data.split('_')
        cancer, mutemode = cancer_mutemode.split('-')

        assert mutemode in ['amp', 'del', 'mutate']
        files = [f'{feat_path}/{feat_origin}/{project}_{cancer}']
        if cancer=='brca':
            files = [ [f'{feat_path}/{feat_origin}/{project}_brcad',
                        f'{feat_path}/{feat_origin}/{project}_brcal']]
    
    return files

def data_to_label_f5(project, data,  split, tcga_fold, data_base='', tile_version='mpp0.5'):
    label_dir = DSMIL_CANCER_SUBTYPE_DIR[tile_version]
    genelabel_dir = DSMIL_CANCER_GENE_FILEDIR[tile_version]
    # genelabel_dir2 = DSMIL_CANCER_GENE_FILEDIR_MORE[tile_version]
    dir1 = '/gene_mpp0.5_more/'
    dir0 = '/gene_mpp0.5/'
    assert project in ['tcga', 'dfciv2']
    if data in _CANCER_SUBTYPE_DATA:
        data1, data2 = data.split('-')
        if project == 'tcga':
            assert split and tcga_fold
            files = [f'{label_dir}/{project}_{data1}diag/f{tcga_fold}_{split}.csv',
                     f'{label_dir}/{project}_{data2}diag/f{tcga_fold}_{split}.csv']
        else:
            files = [f'{label_dir}/{project}_{data1}.csv',
                     f'{label_dir}/{project}_{data2}.csv']
        if data_base:
            files = [f.replace(BASE_DIR, data_base) for f in files]
    else:
        cancer_mutemode, gene = data.split('_')
        cancer, mutemode = cancer_mutemode.split('-')
        
        assert mutemode in ['amp', 'del', 'mutate']

        genefile = f'{genelabel_dir}/{project}_{cancer}_{mutemode}/{gene}'
        if data_base:
            genefile = genefile.replace(BASE_DIR, data_base)
            # files = [f.replace(BASE_DIR, data_base) for f in files]
        
        if project == 'tcga' and split and tcga_fold:
            files = [f'{genefile}/f{tcga_fold}_{split}.csv']
        else:
            files = [f'{genefile}.csv']
        if not all([os.path.exists(f) for f in files]):
            files = [genefile.replace(dir0, dir1) for genefile in files]
            assert all([os.path.exists(f) for f in files]), files
    return files

def data_to_prompts(data):
    if data in _CANCER_SUBTYPE_DATA:
        data1, data2 = data.split('-')
        prompts = [_CANCER_SUBTYPE_DEF[data1],
                   _CANCER_SUBTYPE_DEF[data2]]
    else:
        # cancer, mutemode, gene = data.split('-')
        cancer_mutemode, gene = data.split('_')
        cancer, mutemode = cancer_mutemode.split('-')
        assert mutemode in ['amp', 'del', 'mutate']
        mutate_types = _MUTATE_TYPES[mutemode]
        cancer_def = _CANCER_SUBTYPE_DEF[cancer]
        prompts = [
            f'{gene} gene {mtype} in {cancer_def}' for mtype in mutate_types
        ]
    return prompts



class BagFeatLabelDatsetWSI(Dataset):
    def __init__(self, bag_max_tile, wsifeat_path_list, wsi_label_list, prompt_tokens, wsi_slideid, wsi_patientid, wsi_site, shuffle_patch=True):
        assert len(wsifeat_path_list) == len(wsi_label_list) == len(wsi_patientid) == len(wsi_site) == len(wsi_slideid)
        self.wsifeat_path_list = wsifeat_path_list
        self.wsi_slideid = wsi_slideid
        self.wsi_patientid = wsi_patientid
        self.wsi_site = wsi_site
        self.wsi_label_list = wsi_label_list
        self.shuffle_patch = shuffle_patch
        self.init_label_cnt(wsi_label_list)
        if prompt_tokens:
            self.prompt_tokens = prompt_tokens 
        
        self.num_wsi = len(wsifeat_path_list)
        self.bag_max_tile = bag_max_tile

    def __len__(self):
        return self.num_wsi
    
    def init_label_cnt(self, wsi_label_list):
        pos_count = sum(wsi_label_list)
        self.pos_weight = (len(wsi_label_list)-pos_count)/pos_count

    def get_feats(self, bag_feat_path):
        if bag_feat_path.endswith('.npy'):
            feats = np.load(bag_feat_path)
            return feats
        else:   
            featsall = read_hdf5(bag_feat_path, idx=None)
            num_tile = featsall.shape[0]
            if num_tile >= self.bag_max_tile:
                if self.shuffle_patch:
                    selected_idx =  random.sample(range(num_tile), self.bag_max_tile)
                else:
                    selected_idx =  range(self.bag_max_tile)
            else: # bs=1, no need to fill
                if self.shuffle_patch:
                    selected_idx = random.sample(range(num_tile), num_tile)
                else:
                    selected_idx = range(num_tile)
            
            return featsall[selected_idx]

    def __getitem__(self, idx):
        # bag idx
        bag_feat_path = self.wsifeat_path_list[idx]
        feats = self.get_feats(bag_feat_path)
        label = self.wsi_label_list[idx]
        
        return feats, label, self.wsi_slideid[idx]
    
def cut_cancer_prob_indices(h5f, cancer_prob_thresh):
    prob = read_hdf5(h5f)
    
    selected_idx = np.where(prob>cancer_prob_thresh)[0]
    return selected_idx.tolist()

class MixBagFeatLabelDatsetWSI(BagFeatLabelDatsetWSI):
    def __init__(self, bag_max_tile, wsifeat_path_list, wsi_label_list, prompt_tokens, wsi_slideid, wsi_patientid, wsi_site, cancer_prob_thresh=0, shuffle_patch=True):
        super().__init__(bag_max_tile, wsifeat_path_list, wsi_label_list, prompt_tokens, wsi_slideid, wsi_patientid, wsi_site)
        assert all(isinstance(path, dict) for path in self.wsifeat_path_list)
        self.cancer_prob_thresh = cancer_prob_thresh
        assert self.cancer_prob_thresh >= 0 and self.cancer_prob_thresh <= 0.5, self.cancer_prob_thresh
        self.shuffle_patch = shuffle_patch
        print(f'cancer_prob_thresh: {self.cancer_prob_thresh}..Shuffle:{shuffle_patch}..Max:{bag_max_tile}')

    def get_feats(self, bag_feat_path_dict):
        featdict = {}
        if self.cancer_prob_thresh>0:
            index_pool = cut_cancer_prob_indices(bag_feat_path_dict['ours0_canprob'], self.cancer_prob_thresh)
        else:
            index_pool = []
        for key, bag_feat_path in bag_feat_path_dict.items():
            if key.endswith('_canprob'):
                continue
            featsall = read_hdf5(bag_feat_path, idx=None)
            if len(index_pool)>0:
                if self.shuffle_patch:
                    num_tile = len(index_pool)
                    selected_idx = random.sample(index_pool, num_tile)
                else:
                    selected_idx = index_pool
            else:
                num_tile = featsall.shape[0]
                if self.shuffle_patch:
                    if num_tile >= self.bag_max_tile:
                        selected_idx =  random.sample(range(num_tile), self.bag_max_tile)
                    else:
                        selected_idx = random.sample(range(num_tile), num_tile)
                else:
                    if num_tile >= self.bag_max_tile:
                        selected_idx =  range(self.bag_max_tile)
                    else:
                        selected_idx = range(num_tile)
            featdict[key] = featsall[selected_idx]
        return featdict

def load_feat_label_from_subtype_file(label_file, feat_origin, feat_dir, label, cancer_prob_thresh=0):
    df = pandas.read_csv(label_file)
    if label == None:
        slide_label = df.label
    else:
        slide_label = len(df)*[label]
    
    if 'site' in df.columns:
        slide_site = df.site
    else:
        slide_site = len(df)*['']
    
    wsi_featpath = []
    wsi_slideid = []
    wsi_patientid = []
    wsi_site = []
    wsi_label = []
    wsi_idx = 0
    for n, (slide_id, patient_id, label, site) in enumerate(zip(df['slide_id'], df['patient_id'], slide_label, slide_site)):
        if isinstance(feat_dir, str):
            feat_path = f'{feat_dir}/{slide_id}.npy' if feat_origin=='prism' else f'{feat_dir}/{slide_id}.h5'
            wsi_slideid += [slide_id]
            wsi_patientid += [patient_id]
            wsi_site += [site]
            wsi_label += [label]
            wsi_idx += 1
            if os.path.exists(feat_path):
                wsi_featpath += [feat_path]
            else:
                assert feat_origin in _MULTI_FEAT.keys(), f'{feat_dir}:{feat_path}'
                feat_keys = _MULTI_FEAT[feat_origin] 
                if cancer_prob_thresh:
                    feat_keys += [f'ours0_canprob']
                feat_path_dict = {}
                for key in feat_keys:
                    feat_path_dict[key] = feat_path.replace(f'/{feat_origin}/', f'/{key}/')
                wsi_featpath += [feat_path_dict]                
        else:
            assert isinstance(feat_dir, list)
            feat_path = ''

            for fdir in feat_dir:
                fpath = f'{fdir}/{slide_id}.h5' if not feat_origin=='prism' else f'{fdir}/{slide_id}.npy'
                if feat_origin in _MULTI_FEAT.keys():
                    feat_keys = _MULTI_FEAT[feat_origin]
                    if cancer_prob_thresh:
                        feat_keys += [f'ours0_canprob']
                    feat_path_dict = {}
                    for key in feat_keys:
                        feat_path_dict[key] = fpath.replace(f'/{feat_origin}/', f'/{key}/')
                    if all([os.path.exists(f) for f in feat_path_dict.values()]):
                        feat_path = feat_path_dict
                        break
                else:
                    if os.path.exists(fpath):
                        feat_path = fpath
                        break
                    
                    
            assert feat_path, feat_path
            wsi_featpath += [feat_path]
            wsi_slideid += [slide_id]
            wsi_patientid += [patient_id]
            wsi_site += [site]
            wsi_label += [label]
            wsi_idx += 1
   
    return wsi_featpath, wsi_slideid, wsi_patientid, wsi_site, wsi_label


def load_wsi_feat_prompt_dataset(feat_origin, data_list, split, tcga_fold='', project='tcga', max_tiles_bag_mil=0, data_base='', cancer_prob_thresh=0, shuffle_patch=True):
    if project == 'tcga':
        assert split in ['train', 'val', 'test']
        assert tcga_fold in ['1', '2', '3', '4', '5']
    else:
        assert split in ['']
        assert tcga_fold in ['']

    all_label_prompts = []
    all_wsi_feat_paths = []
    all_wsi_label = []
    all_wsi_slideid = []
    all_wsi_patientid = []
    all_wsi_site = []
    n_unique_label = 0
    for data in data_list:
        label_prompts = data_to_prompts(data)
        all_label_prompts += label_prompts

        label_files = data_to_label_f5(project = project,
                                        data = data, 
                                        split = split,
                                        tcga_fold = tcga_fold,
                                        data_base = data_base)

        feat_files = data_to_feat_dirs(project = project,
                                        data = data,
                                        feat_origin = feat_origin,
                                        data_base=data_base)
        if len(data_list)==1 and len(label_files)==1:
            use_unique_label = False
        else:
            use_unique_label = True
        
        for label_file, feat_file in zip(label_files, feat_files):
            wsi_featpath, wsi_slideid, wsi_patientid, wsi_site, wsi_label = load_feat_label_from_subtype_file(
                                                                    label_file = label_file,
                                                                    feat_origin = feat_origin,
                                                                    feat_dir = feat_file, 
                                                                    label = n_unique_label if use_unique_label else None, 
                                                                    cancer_prob_thresh=cancer_prob_thresh)
            all_wsi_feat_paths += wsi_featpath
            all_wsi_label += wsi_label
            all_wsi_slideid += wsi_slideid
            all_wsi_patientid += wsi_patientid
            all_wsi_site += wsi_site
            n_unique_label += 1 
    

    prompt_tokens = None

    kwargs = {'wsifeat_path_list':all_wsi_feat_paths,
            'wsi_label_list': all_wsi_label,
            'prompt_tokens': prompt_tokens,
            'wsi_slideid': all_wsi_slideid,
            'wsi_patientid':all_wsi_patientid,
            'wsi_site':all_wsi_site,
            'shuffle_patch': shuffle_patch,}
    if feat_origin in cfg.MULTI_FEAT.keys():
        kwargs.update({'cancer_prob_thresh': cancer_prob_thresh})
        dataset = MixBagFeatLabelDatsetWSI(bag_max_tile=max_tiles_bag_mil, **kwargs)
    else:
        assert cancer_prob_thresh == 0
        dataset = BagFeatLabelDatsetWSI(bag_max_tile=max_tiles_bag_mil, **kwargs)
    num_slides = len(dataset)
    logging.info(f'{num_slides} WSI')

    return dataset


def load_wsi_feat_prompt_dataset_dict(feat_origin, dataname_list, tcga_fold='', max_tiles_bag_mil=0, data_base='', cancer_prob_thresh=0, shuffle_patch=True):
    dataset_dict = {}
    

    for dataname in dataname_list:
        split = ''
        if len(dataname.split('_'))==2:
            project, data = dataname.split('_')
        elif len(dataname.split('_'))==3:
            if dataname.endswith(('train', 'val', 'test')):
                project, data, split = dataname.split('_')
            else:
                project, data, gene = dataname.split('_')
                data = '_'.join([data, gene])
        elif len(dataname.split('_'))==4:
            project, data, gene, split = dataname.split('_')
            data = '_'.join([data, gene])
        else: 
            raise ValueError
        logging.info(project, data, split)

        dataset = load_wsi_feat_prompt_dataset(feat_origin=feat_origin,
                                                data_list=[data],
                                                split=split,
                                                tcga_fold=tcga_fold if project=='tcga' else '',
                                                project=project,
                                                max_tiles_bag_mil=max_tiles_bag_mil,
                                                data_base=data_base,
                                                cancer_prob_thresh=cancer_prob_thresh,
                                                shuffle_patch=shuffle_patch)
        dataset_dict[dataname] = dataset
    return dataset_dict