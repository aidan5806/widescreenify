from PIL import Image
import numpy as np
import cv2
import os, shutil
import ffmpeg 

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
        shifted_data[:, dx:] = 255
    elif dx > 0:
        shifted_data[:, 0:dx] = 255

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = 255
    elif dy > 0:
        shifted_data[0:dy, :] = 255
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

def create_video_for_frames(frame_dir, fps, output_path):
    glob_pttn = os.path.join(frame_dir, '*.jpg')
    (ffmpeg.input(glob_pttn, pattern_type='glob', framerate=fps)
        .output(output_path)
        .run())

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
    video_dir = clean_or_mk_data_dir('video', name, force = force)

    print(f'Frame Dir: {frame_dir}')
    print(f'Video Dir: {video_dir}')
    
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
        
        saved_frame = frame[:,:, ::-1].astype(np.uint8)

        saved_frame_width = len(saved_frame[0])
        if saved_frame_width % 2 == 1: 
            saved_frame = saved_frame[:,0:(saved_frame_width-1),:]

        saved_frame_height = len(saved_frame)
        if saved_frame_height % 2 == 1: 
            saved_frame = saved_frame[0:(saved_frame_height-1),:, :]
            
        num = str(step).ljust(5, '0')
        frame_fpath = os.path.join(frame_dir, f'{name}_{num}.jpg')
        Image.fromarray(saved_frame, 'RGB').save(frame_fpath)



    video_name = os.path.join(video_dir, f'{name}.mp4')
    create_video_for_frames(frame_dir, fps, video_name)


## Example Usage ##
video_cfg = {
    'name': 'test', 
    'height': 100, 
    'width': 100, 
    'secs': 4, 
    'foreground': {
        'nodes': [
            cfg_generators.circle_cfg((0,0), 50)
        ],
        'movement': {
            'dx': 1, 
            'dy': 0, 
            'y_wraps': False, 
            'x_wraps': False, 
        },   
    }
}
make_video(**video_cfg)