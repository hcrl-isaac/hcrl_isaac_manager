"""Stack equal-length clips side by side with a label on each, into one mp4.

usage: stack_clips.py OUT.mp4 LABEL=CLIP.mp4 [LABEL=CLIP.mp4 ...] [--scale 0.5]
"""

import argparse
import numpy as np

import imageio.v2 as iio
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("clips", nargs="+")
ap.add_argument("--scale", type=float, default=0.5)
args = ap.parse_args()
readers = [(spec.split("=", 1)[0], iio.get_reader(spec.split("=", 1)[1])) for spec in args.clips]
fps = readers[0][1].get_meta_data().get("fps", 50)
with iio.get_writer(args.out, fps=fps, codec="libx264", quality=8, macro_block_size=8) as w:
    for frames in zip(*(r for _, r in readers), strict=False):
        panels = []
        for (label, _), f in zip(readers, frames, strict=False):
            img = Image.fromarray(np.asarray(f))
            img = img.resize((int(img.width * args.scale), int(img.height * args.scale)))
            ImageDraw.Draw(img).text((10, 8), label, fill=(255, 255, 255))
            panels.append(np.asarray(img))
        w.append_data(np.concatenate(panels, axis=1))
print("wrote", args.out)
