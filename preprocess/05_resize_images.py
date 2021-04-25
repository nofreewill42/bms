from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
cpus = multiprocessing.cpu_count()
cpus = min(48,cpus)

ds_path_str = Path('data_path.txt').read_text()
ds_path = Path(ds_path_str)
PATH = ds_path/'images'                  # input
DEST = ds_path/'images'/'resized'        # output (DEST/size/{train,test})
szs = ((256, 2), )                   # size(s)
resample_type = Image.BICUBIC   # type of resampling

def resize_img(im, fn, sz):
    new_fn = DEST/f'{sz[0]}_{sz[1]}'/fn.relative_to(PATH)
    new_fn.parent.mkdir(parents=True, exist_ok=True)
    w, h = im.size
    wM, hM = sz[0], int(sz[0]/sz[1])
    ratio = max(w/wM, h/hM)
    if ratio>1:
        if new_fn.exists():
            return
        im = im.resize((int(w/ratio), int(h/ratio)), resample=resample_type)
        im.save(new_fn)
    else:
        try: new_fn.unlink()
        finally: new_fn.symlink_to(fn)

def resizes(fn):
    im = Image.open(fn)
    for sz in szs: resize_img(im, fn, sz)

def resize_imgs(p):
    files = p.glob('[0-9a-f]/[0-9a-f]/[0-9a-f]/*.png')
    with ProcessPoolExecutor(cpus) as e: e.map(resizes, files)

# Do the job
for ds in ('train',): resize_imgs(PATH/ds)

