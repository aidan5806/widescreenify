import os
from turtle import Turtle
from tempfile import TemporaryDirectory
from PIL import Image

import cv2
import numpy as np



def convert_to_frame(turtle: Turtle) -> np.ndarray: 
    with TemporaryDirectory() as tmp_dir:
        # Save file to postscript and use pillow to convert
        fpath = os.path.join(tmp_dir, 'frame.ps')
        turtle.getscreen().getcanvas().postscript(file=fpath)
        ps_img = Image.open(fpath)
        
        png_fpath = os.path.join(tmp_dir, 'frame.png')
        ps_img.save(png_fpath)
        img_arr = cv2.imread(png_fpath)
    
    return img_arr


def create_foreground(frame_config: dict, height: int, width: int) -> np.ndarray: 
    # Draw the nodes indicated by the config
    turtle = Turtle() 
    turtle.getscreen().screensize(canvwidth=width, canvheight=height)
    turtle.hideturtle() 
    turtle.speed('fastest')

    nodes = frame_config['nodes']
    for node_cfg in nodes: 
        gen_fn = node_cfg.pop('generator_fn')
        start = node_cfg.pop('start')
        gen_fn(turtle, start, **node_cfg)

    return convert_to_frame(turtle)