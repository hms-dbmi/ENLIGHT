import os, glob
import shutil
from PIL import Image
from utils.constants import CACHE_DIR
from preprocess.slide_tile_utils import get_args, DeepZoomStaticTilerMPP, tiles_to_hdf5
from pathlib import Path 
Image.MAX_IMAGE_PIXELS = None

def crop_slide_mpp(slide_path, slide_patch_path, desired_mpp, cache_dir=CACHE_DIR,
                   tile_size=224, overlap=0, fmt='jpeg', quality=70, workers=10,
                   background_t=15, defaultmpp=0.5):
    os.makedirs(cache_dir, exist_ok=True)
    kwargs = {'slidepath': slide_path,
                'basename': cache_dir,
                'mag_levels': (0,),
                'format': fmt,
                'tile_size': tile_size,
                'overlap': overlap,
                'limit_bounds': True,
                'quality': quality,
                'workers': workers,
                'threshold': background_t}

    DeepZoomStaticTilerMPP(base_mpp=desired_mpp, objective=defaultmpp, **kwargs).run()

    scale_dirs = sorted(glob.glob(f'{cache_dir}/[0-9]*'),
                        key=lambda p: int(os.path.basename(p)))
    assert len(scale_dirs) == 1, len(scale_dirs)
    tiles_to_hdf5(scale_dirs[0], slide_patch_path, tile_size=tile_size, ext=fmt)

    shutil.rmtree(scale_dirs[0])


if __name__ == '__main__':
    args = get_args()
    levels = tuple(sorted(args.magnifications))

    svspath = args.input_slide_path
    slideid = Path(svspath).name.split('.')[0]
    # Always tile into a temp folder; convert to HDF5 then delete it.
    # nested_patches / tile_dir are not needed when the target is HDF5.
    temp_folder = f'{args.output_dir}/temp_patch'
    os.makedirs(temp_folder, exist_ok=True)
    kwargs = {'slidepath': svspath,
                'basename': temp_folder,
                'mag_levels': levels,
                'format': args.format,
                'tile_size': args.tile_size,
                'overlap': args.overlap,
                'limit_bounds': True,
                'quality':args.quality,
                'workers': args.workers,
                'threshold': args.background_t}

    DeepZoomStaticTilerMPP(base_mpp= args.mpp, objective=args.defaultmpp, **kwargs).run()


    scale_dirs = sorted(glob.glob(f'{temp_folder}/[0-9]*'),
                        key=lambda p: int(os.path.basename(p)))
    for scale_dir in scale_dirs:
        scale_idx = os.path.basename(scale_dir)
        suffix = f'_scale{scale_idx}' if len(scale_dirs) > 1 else ''
        h5_path = f'{args.output_dir}/{slideid}{suffix}.h5'
        tiles_to_hdf5(scale_dir, h5_path, tile_size=args.tile_size, ext=args.format)

    shutil.rmtree(temp_folder)