from PIL import Image
import numpy as np
import cv2
import os, shutil

from utils import create_foreground
import cfg_generators


DATA_DIR = os.path.join(os.getcwd(), 'data')

axis_map = {
    'x': 0, 'y': 1,
}

def shift_2d_non_replace(data, dx, dy): 
    shifted_data = np.roll(data, dx, axis = 1)
    return np.roll(shifted_data, dy, axis = 0)

# See https://stackoverflow.com/questions/2777907/python-numpy-roll-with-padding
def shift_2d_replace(data, dx, dy):
    """
    Shifts the array in two dimensions while setting rolled values to constant
    :param data: The 2d numpy array to be shifted
    :param dx: The shift in x
    :param dy: The shift in y
    :param constant: The constant to replace rolled values with
    :return: The shifted array with "constant" where roll occurs
    """
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = 0
    elif dx > 0:
        shifted_data[:, 0:dx] = 0

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = 0
    elif dy > 0:
        shifted_data[0:dy, :] = 0
    return shifted_data

def shift_2d(data, dx, dy, x_wraps, y_wraps): 
    x_shift_fn = shift_2d_non_replace if x_wraps else shift_2d_replace
    y_shift_fn = shift_2d_non_replace if y_wraps else shift_2d_replace

    shifted_data = x_shift_fn(data, dx, 0)
    return y_shift_fn(shifted_data, 0, dy)

def video_for_frames(video_name, frame_dir):

    frames = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(frame_dir, frames[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in frames:
        video.write(cv2.imread(os.path.join(frame_dir, image)))

    cv2.destroyAllWindows()
    video.release()

def clean_or_mk_data_dir(subdir, name, force = False):
    subdir_path = os.path.join(DATA_DIR, subdir)
    dir_path = os.path.join(subdir_path, name)
    if os.path.isdir(dir_path) and not force: 
        return 

    if not os.path.isdir(subdir_path):
        os.mkdir(subdir_path)
    
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    
    os.mkdir(dir_path)

    return dir_path

def create_video_for_frames():
    pass

def make_video(
    name: str,
    height: int, 
    width: int, 
    secs: int, 
    foreground: dict = {},
    background: dict = {}, 
    fps: int = 30, 
    force: bool = True
):
    frame_dir = clean_or_mk_data_dir('frames', name, force = force)
    
    movement = foreground.pop('movement', {})
    dx, dy = movement.pop('dx', 0), movement.pop('dy', 0)
    x_wraps, y_wraps = movement.pop('x_wraps', False), movement.pop('x_wraps', False)

    initial_frame = create_foreground(foreground, height, width)
    
    
    frame_for_step = lambda step: shift_2d(
        initial_frame,
        dx * step, 
        dy * step, 
        x_wraps, 
        y_wraps
    )
    

    for step in range(fps * secs):
        frame = frame_for_step(step)
        frame_fpath = os.path.join(frame_dir, f'{name}_{int(step)}.png')
        Image.fromarray(frame[:,:, ::-1].astype(np.uint8), 'RGB').save(frame_fpath)

## Example Usage ##
# video_cfg = {
#     'name': 'test', 
#     'height': 100, 
#     'width': 100, 
#     'secs': 2, 
#     'foreground': {
#         'nodes': [
#             cfg_generators.circle_cfg((0,0), 50)
#         ],
#         'movement': {
#             'dx': 1, 
#             'dy': 1, 
#             'y_wraps': False, 
#             'x_wraps': False, 
#         },   
#     }
# }
# make_video(**video_cfg)