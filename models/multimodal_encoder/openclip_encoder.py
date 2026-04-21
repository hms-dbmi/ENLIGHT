import torch
import torch.nn as nn
import open_clip
from datasets.slide_dataset import LoadTileH5Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils.io import read_hdf5, read_hdf5_size, write_hdf5, init_hdf5_bag, add_hdf5_bag

class OpenCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower_path, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_path = vision_tower_path  # path to .pt checkpoint
        self.model_type = getattr(args, 'mm_vision_tower_model_type', 'ViT-L-14-336')
        self.cache_dir = getattr(args, 'cache_dir', None)

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            print('Model loading delayed.')

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_path))
            return
        device = device_map if isinstance(device_map, str) and device_map != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        model, _, image_processor = open_clip.create_model_and_transforms(
            self.model_type,
            pretrained=self.vision_tower_path,
            cache_dir=self.cache_dir,
            force_quick_gelu=True, 
            device=device
        )
        self.clip = model
        self.vision_tower = model.visual  # only need the vision encoder
        self.vision_tower.output_tokens = True
        self.image_processor = image_processor
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        """
        Returns patch token features: (B, N_patches, hidden_size)
        Uses open_clip's output_tokens=True to get per-patch features
        from the final transformer layer (before projection head).
        """
        
        if type(images) is list:
            image_features = []
            for image in images:
                y = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
                image_feature = y[1]
                image_features.append(image_feature.to(image.dtype))
        else:
            y = self.vision_tower(images.to(device=self.device, dtype=self.dtype)) ##(1, 1024), (576, 1024)
            image_features = y[1]# drop CLS token
            image_features = image_features.to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return self.vision_tower

    @property
    def hidden_size(self):
        # ViT-L-14 hidden dim
        return self.vision_tower.transformer.width

    @property
    def num_patches_per_side(self):
        return self.vision_tower.image_size[0] // self.vision_tower.patch_size[0]

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2

class OpenCLIPEnc():
    def __init__(self,
                 model,
                 image_processor,
                 pretrained_path = '',
                 cache_dir = './cache',
                 model_type='ViT-L-14-336', 
                 imgfeat_type='cls_patch'):
        self.tokenizer = open_clip.get_tokenizer(model_type)
        if model and image_processor:
            self.model = model
            self.image_processor = image_processor
        else:
            self.model, _, self.image_processor = open_clip.create_model_and_transforms(model_type,
                                                                        pretrained=pretrained_path,
                                                                        cache_dir=cache_dir,
                                                                        force_quick_gelu=True)
        self.device = 'cuda' if torch.cuda.device_count()>0 else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.visual.output_tokens = True
        self.imgfeat_type = imgfeat_type
        self.logit_exp = self.model.logit_scale.exp().item()

    def encode_text(self, text_list, norm=False):
        with torch.no_grad():
            token_ids = self.tokenizer(texts = text_list).to(self.device)
            text_embeddings = self.model.encode_text(token_ids)

        if norm:
            text_embeddings /= text_embeddings.norm(dim=1, keepdim=True)

        return text_embeddings

    def encode_images(self, imgs, imgfeat_type=None):
        if imgfeat_type is None:
            imgfeat_type = self.imgfeat_type
        with torch.no_grad():
            y = self.model.visual(imgs.to(self.device))
            if imgfeat_type == 'cls_patch': #(1, 1024)
                return y[0]
            elif imgfeat_type == 'patch': #(576, 1024)
                return y[1]
            elif imgfeat_type == 'both':
                return y[0], y[1]


    def cosine_similarity(self, text_embedding_1, text_embedding_2, img_embedding):
        sim1 = img_embedding @ text_embedding_1.T #(BS, D) @ (D, num_text) -> (BS, num_text)
        sim1 = sim1.mean(dim=-1)
        sim2 = img_embedding @ text_embedding_2.T #(BS, D) @ (D, num_text) -> (BS, num_text)
        sim2 = sim2.mean(dim=-1)

        similarity = torch.stack([sim1, sim2], dim=-1)
        prob = (self.logit_exp * similarity).softmax(dim=-1) 
        return prob
    

    def extract_slide_feature(self, slide_patch_path):
        dataset = LoadTileH5Dataset(path=slide_patch_path, preprocess=self.image_processor, is_hf_processor=False)
        dataloader = DataLoader(dataset,
                        batch_size=256 if self.device=='cuda' else 16,
                        num_workers=2,
                        shuffle=False,
                        drop_last=False)
        for imgs, coords in tqdm(dataloader):
            self.encode_images(imgs, imgfeat_type='cls_patch')

    def cancerous_patch_filter(self, slide_patch_path, slide_all_feats, cancer_prob_thresh, cancerous_slide_patch_path=''):
        text_emb1 = self.encode_text(text_list=['cancer', 'cancerous', 'tumor'])
        text_emb2 = self.encode_text(text_list=['normal', 'non-cancerous', 'healthy', 'benign'])

        dataset = LoadTileH5Dataset(path=slide_patch_path, preprocess=self.image_processor, is_hf_processor=False)
        dataloader = DataLoader(dataset,
                        batch_size=256 if self.device=='cuda' else 16,
                        num_workers=2,
                        shuffle=False,
                        drop_last=False)
        
        slide_cancer_prob_pass = []
        slide_all_coords = []
        for imgs, coords in tqdm(dataloader):
            img_embedding = self.encode_images(imgs, imgfeat_type='cls_patch')
            img_embedding /= img_embedding.norm(dim=1, keepdim=True)
            sims = self.cosine_similarity(text_emb1, text_emb2, img_embedding) #(Bs, 2)
            cancer_prob_pass = (sims[:, 0] > cancer_prob_thresh).cpu().numpy()
            slide_cancer_prob_pass.append(cancer_prob_pass)
            slide_all_coords.append(coords.numpy())
        slide_cancer_prob_pass = np.concatenate(slide_cancer_prob_pass)
        slide_all_coords = np.concatenate(slide_all_coords)
        # Filter slide feats
        slide_filtered_feats = {}
        for key, feats in slide_all_feats.items():
            _feats = feats[slide_cancer_prob_pass]
            slide_filtered_feats[key] = _feats
        # Filter slide patch
        slide_filtered_coords = slide_all_coords[slide_cancer_prob_pass]
        if cancerous_slide_patch_path:
            try:
                slide_patch = read_hdf5(slide_patch_path)
                slide_filtered_patch = slide_patch[slide_cancer_prob_pass]
                write_hdf5(cancerous_slide_patch_path, slide_filtered_patch, coords=slide_filtered_coords)
            except:
                print('Sequetially processing....')
                chunk_size = 256
                total = read_hdf5_size(slide_patch_path)[0]
                pass_indices = np.where(slide_cancer_prob_pass)[0]
                initialized = False
                filtered_so_far = 0
                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    local_idx = pass_indices[(pass_indices >= start) & (pass_indices < end)] - start
                    if len(local_idx) == 0:
                        continue
                    n = len(local_idx)
                    chunk = read_hdf5(slide_patch_path, idx=slice(start, end))[local_idx]
                    coord_chunk = slide_filtered_coords[filtered_so_far:filtered_so_far + n]
                    filtered_so_far += n
                    if not initialized:
                        init_hdf5_bag(cancerous_slide_patch_path, chunk, coord_x_y=coord_chunk)
                        initialized = True
                    else:
                        add_hdf5_bag(cancerous_slide_patch_path, chunk, coord_x_y=coord_chunk)

        return slide_filtered_feats