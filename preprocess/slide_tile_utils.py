"""
Code adapated from https://github.com/binli123/dsmil-wsi/blob/master/deepzoom_tiler.py
"""

from multiprocessing import Process, JoinableQueue
import argparse
import os
import re
import shutil
import sys
import glob
import numpy as np
import math
from unicodedata import normalize
from PIL import Image, ImageFilter, ImageStat
from tqdm import tqdm
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
import h5py

from utils.constants import CACHE_DIR

VIEWER_SLIDE_NAME = 'slide'

class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
                quality, threshold):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._threshold = threshold
        self._slide = None

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            # print(outfile)
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            try:
                tile = dz.get_tile(level, address)
                edge = tile.filter(ImageFilter.FIND_EDGES)
                edge = ImageStat.Stat(edge).sum
                edge = np.mean(edge)/(self._tile_size**2)
                w, h = tile.size
                if edge > self._threshold:
                    if not (w==self._tile_size and h==self._tile_size):
                        tile = tile.resize((self._tile_size, self._tile_size))
                    tile.save(outfile, quality=self._quality)
            except:
                pass
            self._queue.task_done()
            

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)

class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, target_levels, format, associated, queue):
        self._dz = dz
        self._basename = basename
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._target_levels = target_levels
        
    def run(self):
        self._write_tiles()

    def _write_tiles(self):
        target_levels = [self._dz.level_count-i-1 for i in self._target_levels] #[3,1] -> [18-3-1, 18-1-1]=[14, 16] # higher level is finer 
        # mag_list = [int(self._mag_base/2**i) for i in self._target_levels] #[3,1] -> naming the temp file -> to organize pyramid 
        
        scale_idx = 0
        print('Levels', self._dz.level_count, self._target_levels)
        for level in range(self._dz.level_count): #
            if not (level in target_levels):
                continue
            tiledir = os.path.join("%s" % self._basename, str(scale_idx))
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)

            cols, rows = self._dz.level_tiles[level]
            print("Overall #tiles:", level, cols*rows, tiledir)
            # print(self._dz.level_tiles[level], self._dz.level_tiles[level-1])
    
            for row in tqdm(range(rows)):
                for col in tqdm(range(cols)):
                    tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                    col, row, self._format))
                    if not os.path.exists(tilename):
                        self._queue.put((self._associated, level, (col, row),
                                    tilename))
                    self._tile_done()
            scale_idx += 1

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    self._associated or 'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)

class DeepZoomStaticTilerMAG(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, mag_levels, base_mag, objective, format, tile_size, overlap,
                limit_bounds, quality, workers, threshold):
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._mag_levels = mag_levels
        self._base_mag = base_mag
        self._objective = objective
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._dzi_data = {}
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                        limit_bounds, quality, threshold).start()

    def run(self):
        self._run_image()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)
        
        MAG_BASE = self._slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER) #40 
        if MAG_BASE is None:
            MAG_BASE = self._objective
        first_level = int(math.log2(float(MAG_BASE)/self._base_mag)) # raw / input, 40/20->1, 40/40->0
        target_levels = [i+first_level for i in self._mag_levels] # levels start from 0, [0,2]->[1,3]
        target_levels.reverse()
        print(target_levels)
        tiler = DeepZoomImageTiler(dz, basename, target_levels, self._format, associated,
                    self._queue)
        tiler.run()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()

class DeepZoomStaticTilerMPP(DeepZoomStaticTilerMAG):
    def __init__(self, slidepath, basename, mag_levels, base_mpp, objective, format, tile_size, overlap,
                limit_bounds, quality, workers, threshold):
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._mag_levels = mag_levels
        self._base_mpp = base_mpp
        self._objective = objective
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._dzi_data = {}
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                        limit_bounds, quality, threshold).start()
            
    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)
        

        MPP_BASE = self._slide.properties.get('openslide.mpp-x')
        if MPP_BASE is None:
            MPP_BASE = self._objective
        
        # first_level = int(math.log2(float(MAG_BASE)/self._base_mag))  raw / input, 40/20->1, 40/40->0
        first_level = round(math.log2(self._base_mpp / float(MPP_BASE)))
        target_levels = [i+first_level for i in self._mag_levels]
        target_levels.reverse()
        print('MPP', MPP_BASE, self._base_mpp, target_levels, dz.level_tiles[0])
        tiler = DeepZoomImageTiler(dz, basename, target_levels, self._format, associated,
                    self._queue)
        tiler.run()

class DeepZoomImageDownsizeTiler(DeepZoomStaticTilerMPP):
    def __init__(self, slidepath, basename, mag_levels, base_mpp, downsample, format, tile_size, overlap,
                limit_bounds, quality, workers, threshold):
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._mag_levels = mag_levels
        self._base_mpp = base_mpp
        self._downsample = downsample
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._dzi_data = {}
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                        limit_bounds, quality, threshold).start()
    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)
        target_levels = [round(math.log2(self._downsample))] # downsample 2 -> level 1 
        tiler = DeepZoomImageTiler(dz, basename, target_levels, self._format, associated,
                    self._queue)
        tiler.run()

def nested_patches(bag_path, level=(0,), ext='jpeg', tmp_dir='WSI_temp_files'):
    print('\n Organizing patches')
    n_scales = len(glob.glob(f'{tmp_dir}/*'))
    os.makedirs(bag_path, exist_ok=True)
    if len(level)==1:
        patches = glob.glob(os.path.join(tmp_dir, '*', '*.'+ext))
        for i, patch in enumerate(patches):
            patch_name = patch.split(os.sep)[-1]
            shutil.move(patch, os.path.join(bag_path, patch_name))
            sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(patches)))
        print('Done.')
    else:
        level_factor = 2**int(level[1]-level[0])
        levels = [int(os.path.basename(i)) for i in glob.glob(os.path.join(tmp_dir, '*'))]
        levels.sort() # increasing
        low_patches = glob.glob(os.path.join(tmp_dir, str(levels[0]), '*.'+ext))
        for i, low_patch in enumerate(low_patches):
            low_patch_name = low_patch.split(os.sep)[-1]
            shutil.move(low_patch, os.path.join(bag_path, low_patch_name))
            low_patch_folder = low_patch_name.split('.')[0]
            high_patch_path = os.path.join(bag_path, low_patch_folder)
            os.makedirs(high_patch_path, exist_ok=True)
            low_x = int(low_patch_folder.split('_')[0])
            low_y = int(low_patch_folder.split('_')[1])
            high_x_list = list( range(low_x*level_factor, (low_x+1)*level_factor) )
            high_y_list = list( range(low_y*level_factor, (low_y+1)*level_factor) )
            for x_pos in high_x_list:
                for y_pos in high_y_list:
                    high_patch = glob.glob(os.path.join(tmp_dir, str(levels[1]), '{}_{}.'.format(x_pos, y_pos)+ext))
                    if len(high_patch)!=0:
                        high_patch = high_patch[0]
                        shutil.move(high_patch, os.path.join(bag_path, low_patch_folder, high_patch.split(os.sep)[-1]))
            try:
                # os.rmdir(os.path.join(bag_path, low_patch_folder))
                os.remove(low_patch)
            except:
                pass
            sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(low_patches)))
        print('Done.')



def get_args():
    parser = argparse.ArgumentParser(description='Patch extraction for WSI')
    parser.add_argument('-e', '--overlap', type=int, default=0, help='Overlap of adjacent tiles [0]')
    parser.add_argument('-f', '--format', type=str, default='jpeg', help='Image format for tiles [jpeg]')
    parser.add_argument('-j', '--workers', type=int, default=10, help='Number of worker processes to start [10]')
    parser.add_argument('-q', '--quality', type=int, default=70, help='JPEG compression quality [70]')
    parser.add_argument('-s', '--tile_size', type=int, default=224, help='Tile size [224]')
    parser.add_argument('-m', '--magnifications', type=int, nargs='+', default=(0,), help='Levels for patch extraction [0]')
    parser.add_argument('-t', '--background_t', type=int, default=15, help='Threshold for filtering background [15]')
    parser.add_argument('--target', type=str, default='mpp', choices=['mag', 'mpp', 'downsample'],help='use base mag or mpp as target')
    parser.add_argument('--defaultmpp', type=float, default=0.5, help='The default mpp of slide level0 if metadata does not present')
    parser.add_argument('--defaultmag', type=float, default=20, help='The default objective power of slide level0 if metadata does not present')
    parser.add_argument('--mag', type=float, default=20, help='Maximum magnification for patch extraction [level0]')
    parser.add_argument('--mpp', type=float, default=0.5, help='Maximum mpp for patch extraction [level0]')
    parser.add_argument('--downsample', type=float, default=2, help='Downsample factor for patch extraction [level0]')
    parser.add_argument('--input-slide-path', required=True, type=str, help='Input WSI file')
    parser.add_argument('--output-dir', default=CACHE_DIR, type=str)
    args = parser.parse_args()
    return args


def tiles_to_hdf5(tile_dir, h5_path, tile_size=224, ext='jpeg'):
    """
    Pack JPEG tiles from tile_dir into an HDF5 file in row-major order.

    Tile filenames must follow the pattern {col}_{row}.{ext} as produced by
    DeepZoomImageTiler.  Tiles are stored in row-major sequence:
        (row=0,col=0), (row=0,col=1), ..., (row=1,col=0), ...

    HDF5 layout:
        /imgs    uint8  (N, tile_size, tile_size, 3)
        /coords  int32  (N, 2)   columns: [col, row]

    For pyramid output (multiple scale sub-directories named 0, 1, ...),
    call this function once per sub-directory.

    Args:
        tile_dir : directory containing {col}_{row}.{ext} files
        h5_path  : output .h5 file path
        tile_size: expected tile spatial size (used to pre-allocate the array)
        ext      : tile file extension without dot (default 'jpeg')
    """
    pattern = os.path.join(tile_dir, f'*.{ext}')
    tile_paths = glob.glob(pattern)
    if not tile_paths:
        print(f'[tiles_to_hdf5] No tiles found in {tile_dir}')
        return

    def _parse_coord(path):
        name = os.path.splitext(os.path.basename(path))[0]  # 'col_row'
        col, row = name.split('_')
        return int(col), int(row)

    # Sort row-major: primary key = row, secondary key = col
    tile_paths.sort(key=lambda p: (_parse_coord(p)[1], _parse_coord(p)[0]))

    n = len(tile_paths)
    os.makedirs(os.path.dirname(h5_path) if os.path.dirname(h5_path) else '.', exist_ok=True)
    with h5py.File(h5_path, 'w') as f:
        imgs_ds = f.create_dataset(
            'imgs',
            shape=(n, tile_size, tile_size, 3),
            dtype='uint8',
            chunks=(1, tile_size, tile_size, 3),
        )
        coords_ds = f.create_dataset(
            'coords',
            shape=(n, 2),
            dtype='int32',
        )
        for i, path in enumerate(tqdm(tile_paths, desc=f'Writing HDF5 {os.path.basename(h5_path)}')):
            col, row = _parse_coord(path)
            img = np.array(Image.open(path).convert('RGB'))
            if img.shape[0] != tile_size or img.shape[1] != tile_size:
                img = np.array(Image.fromarray(img).resize((tile_size, tile_size)))
            imgs_ds[i] = img
            coords_ds[i] = [col, row]
        f.attrs['n_tiles'] = n
        f.attrs['tile_size'] = tile_size
    print(f'[tiles_to_hdf5] Wrote {n} tiles → {h5_path}')

    
    