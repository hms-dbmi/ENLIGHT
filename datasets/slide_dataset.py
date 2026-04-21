from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.io import read_hdf5

    
class LoadTileH5Dataset(Dataset):
    def __init__(self, path, preprocess, is_hf_processor=False):
        self.h5path = path
        self.preprocess = preprocess
        self.nimgs = read_hdf5(path).shape[0]
        self.is_hf_processor = is_hf_processor

    def __len__(self):
        return self.nimgs
    
    def __getitem__(self, idx):
        image = read_hdf5(self.h5path, idx)
        coords = read_hdf5(self.h5path, idx, key='coords')
        cx, cy = coords
        if self.preprocess is not None:
            if self.is_hf_processor:
                image = self.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
            else:
                image = self.preprocess(Image.fromarray(image))
        return image, torch.tensor([int(cx),int(cy)])
    

    class LoadTileDataset(Dataset):
        def __init__(self, paths, preprocess):
            self.imgpaths = paths
            self.preprocess = preprocess
        
        def __len__(self):
            return len(self.imgpaths)
        
        def __getitem__(self, idx):
            imgpath = self.imgpaths[idx]
            coord_xy = imgpath.split('/')[-1].split('.')[0]
            cx, cy = coord_xy.split('_') 
            image = Image.open(imgpath).convert('RGB')
            if self.preprocess is not None:
                image = self.preprocess(image)
            return image, torch.tensor([int(cx),int(cy)])


