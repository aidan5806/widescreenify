################################################################################
# widescreenify.py
# Aidan Maycock
# Heavily based on a Pix2Pix implementation by Arthur Juliani
# https://github.com/awjuliani/Pix2Pix-Film/blob/master/Pix2Pix.ipynb
################################################################################

import argparse
import os
from datetime import datetime
import ntpath
from sys import platlibdir

import numpy as np
from math import ceil
import random

import tensorflow as tf
import tf_slim as slim

from PIL import Image
import cv2 # https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

# PARAMS
FRAME_GROUP = 1
FRAME_HEIGHT = 144 # 720
FRAME_WIDTH = 256 # 1280
FRAME_RESCALE = 1
EPOCHS = 1
MODEL_SCALE = 32 # Originally 32, used 8 at HD

H_RATIO = 9
W_RATIO = 16
I_RATIO = 12

# GLOBALS
#This initializaer is used to initialize all the weights of the network.
initializer = None

################################################################################
# USER FUNCTIONS
################################################################################

# Check video properties to ensure it can be trained/tested with the model.
def check_props(width, height, mode):
    if (mode == "train"):
        if not ((FRAME_WIDTH == width) and (FRAME_HEIGHT == height)):
            return False
    else:
        if not ((((FRAME_WIDTH // W_RATIO) * I_RATIO) == width) and (FRAME_HEIGHT == height)):
            return False

    return True


# Ingest video and create train and test frame group
# output_frames will return None if mode is test
def get_frame_group(video, current_frame, start_frame, stop_frame, mode, random_frame):
    width_diff = FRAME_WIDTH - ((FRAME_WIDTH // W_RATIO) * I_RATIO)
    width_start = width_diff // 2
    width_end = FRAME_WIDTH - (width_diff // 2)

    if (random_frame):
        current_frame = random.randrange(start_frame, stop_frame-FRAME_GROUP)

    group_stop_frame = current_frame + FRAME_GROUP if ((current_frame + FRAME_GROUP) <= stop_frame) else stop_frame
    group_frame_count = group_stop_frame - current_frame

    video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    for frame_index in range(0, group_frame_count):
        ret, frame = video.read()

        if not ret:
            print("Video read error")
            exit(0)

        if (FRAME_RESCALE):
            if (mode == "train"):
                frame = cv2.resize(frame, dsize=(FRAME_WIDTH,FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
            else:
                frame = cv2.resize(frame, dsize=(((FRAME_WIDTH // W_RATIO) * I_RATIO),FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

        orig_frame = np.copy(frame)
        frame = cv2.GaussianBlur(frame, (7,7), cv2.BORDER_DEFAULT)
        # frame = np.dstack((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

        dom_color_data = np.reshape(frame, (-1,3)).astype(np.float32)

        n_colors = 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, _, palette = cv2.kmeans(dom_color_data, n_colors, None, criteria, 10, flags)

        if (mode == "train"):
            input_frames = np.full((group_frame_count, FRAME_HEIGHT, FRAME_WIDTH, 3), palette[0], np.dtype('float32'))
            output_frames = np.full((group_frame_count, FRAME_HEIGHT, FRAME_WIDTH, 3), palette[0], np.dtype('float32'))

            input_frames[frame_index, :, width_start:width_end, :] = frame[:, width_start:width_end, :]
            output_frames[frame_index] = frame
        else:
            input_frames = np.full((group_frame_count, FRAME_HEIGHT, FRAME_WIDTH, 3), palette[0], np.dtype('float32'))
            output_frames = np.full((group_frame_count, FRAME_HEIGHT, FRAME_WIDTH, 3), palette[0], np.dtype('float32'))

            input_frames[frame_index, :, width_start:width_end, :] = frame
            output_frames[frame_index, :, width_start:width_end, :] = orig_frame

    return group_stop_frame, input_frames, output_frames

#This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.compat.v1.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def generator(c, name):
    with tf.compat.v1.variable_scope('generator_' + name):
        #Encoder
        enc0 = slim.conv2d(c,MODEL_SCALE*2,[3,3],padding="SAME",
            biases_initializer=None,activation_fn=lrelu,
            weights_initializer=initializer)
        enc0 = tf.nn.space_to_depth(enc0,2)

        enc1 = slim.conv2d(enc0,MODEL_SCALE*4,[3,3],padding="SAME",
            normalizer_fn=slim.batch_norm,activation_fn=lrelu,
            weights_initializer=initializer)
        enc1 = tf.nn.space_to_depth(enc1,2)

        enc2 = slim.conv2d(enc1,MODEL_SCALE*4,[3,3],padding="SAME",
            normalizer_fn=slim.batch_norm,activation_fn=lrelu,
            weights_initializer=initializer)
        enc2 = tf.nn.space_to_depth(enc2,2)

        enc3 = slim.conv2d(enc2,MODEL_SCALE*8,[3,3],padding="SAME",
            normalizer_fn=slim.batch_norm,activation_fn=lrelu,
            weights_initializer=initializer)
        enc3 = tf.nn.space_to_depth(enc3,2)

        #Decoder
        gen0 = slim.conv2d(
            enc3,num_outputs=MODEL_SCALE*8,kernel_size=[3,3],
            padding="SAME",normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.elu, weights_initializer=initializer)
        gen0 = tf.nn.depth_to_space(gen0,2)

        gen1 = slim.conv2d(
            tf.concat([gen0,enc2],3),num_outputs=MODEL_SCALE*8,kernel_size=[3,3],
            padding="SAME",normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.elu,weights_initializer=initializer)
        gen1 = tf.nn.depth_to_space(gen1,2)

        gen2 = slim.conv2d(
            tf.concat([gen1,enc1],3),num_outputs=MODEL_SCALE*4,kernel_size=[3,3],
            padding="SAME",normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.elu,weights_initializer=initializer)
        gen2 = tf.nn.depth_to_space(gen2,2)

        gen3 = slim.conv2d(
            tf.concat([gen2,enc0],3),num_outputs=MODEL_SCALE*4,kernel_size=[3,3],
            padding="SAME",normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.elu, weights_initializer=initializer)
        gen3 = tf.nn.depth_to_space(gen3,2)

        g_out = slim.conv2d(
            gen3,num_outputs=3,kernel_size=[1,1],padding="SAME",
            biases_initializer=None,activation_fn=tf.nn.tanh,
            weights_initializer=initializer)
        return g_out


def discriminator(bottom, name, reuse=False):
    with tf.compat.v1.variable_scope('discriminator_' + name):
        filters = [MODEL_SCALE*1,MODEL_SCALE*2,MODEL_SCALE*4,MODEL_SCALE*4]

        #Programatically define layers
        for i in range(len(filters)):
            if i == 0:
                layer = slim.conv2d(bottom,filters[i],[3,3],padding="SAME",scope='d'+str(i),
                    biases_initializer=None,activation_fn=lrelu,stride=[2,2],
                    reuse=reuse,weights_initializer=initializer)
            else:
                layer = slim.conv2d(bottom,filters[i],[3,3],padding="SAME",scope='d'+str(i),
                    normalizer_fn=slim.batch_norm,activation_fn=lrelu,stride=[2,2],
                    reuse=reuse,weights_initializer=initializer)
            bottom = layer

        dis_full = slim.fully_connected(slim.flatten(bottom),MODEL_SCALE*32,activation_fn=lrelu,scope='dl',
            reuse=reuse, weights_initializer=initializer)

        d_out = slim.fully_connected(dis_full,1,activation_fn=tf.nn.sigmoid,scope='do',
            reuse=reuse, weights_initializer=initializer)
        return d_out

################################################################################
# SCRIPT START
################################################################################

# Ingest command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path", type=str, required=True, help="Set path to video file for training or testing.")
parser.add_argument("-o", "--output", type=str, default="./output", help="Output directory for training samples and output videos. Videos will be output with <original_basename>_WIDE.mkv format.")
parser.add_argument("-m", "--mode", type=str, default="train", help="Set to either train or test mode.")
parser.add_argument("-c", "--ckpt", type=str, default="./model", help="Set path to model directory.")
parser.add_argument("-s", "--start_frame", type=int, default=0, help="Set the starting frame for video ingest.")
parser.add_argument("-f", "--stop_frame", type=int, help="Set maximum number of frames to train or test by indicating the stop frame.")
parser.add_argument("-l", "--load_model", action="store_true", help="Load latest checkpoint from model directory. Will default to true if mode is \"test\".")
parser.add_argument("-r", "--random", action="store_true", help="Load a set of frames random point between start frame and stop frame.")

args = parser.parse_args()

video_path = args.path
ckpt_path = args.ckpt
mode = args.mode
stop_frame = args.stop_frame
start_frame = args.start_frame
current_frame = args.start_frame
if (args.load_model or (mode == "test")):
    load_model = True
else:
    load_model = False
output_path = args.output
if (args.random):
    random_frame = True
    random.seed()
else:
    random_frame = False


# Ingest video clip
# Initial Video clip: "D:\HDD Downloads\Media\Avatar - The Legend of Korra\Season 1\The.Legend.of.Korra.S01E01-E02.720p.HDTV.x264-HWE.mkv"
video = cv2.VideoCapture(video_path)

video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # in frames
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = video.get(cv2.CAP_PROP_FPS)

print(f"{frame_width}x{frame_height}, {video_length} frames at {frame_rate}fps")

if (stop_frame != None):
    stop_frame = min(video_length, stop_frame)
else:
    stop_frame = video_length

if (current_frame >= stop_frame):
    print(f"Start frame is after stop frame:: Start:{current_frame} Stop:{stop_frame}")
    exit(1)

if not FRAME_RESCALE:
    if not check_props(frame_width, frame_height, mode):
        print(f"Incompatible input video dimensions:: {frame_width}x{frame_height}")
        exit(1)

################################################################################
# CONNECT MODEL
################################################################################

# Define and connect model
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

#This initializaer is used to initialize all the weights of the network.
initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.02)

condition_in = tf.compat.v1.placeholder(shape=[None,FRAME_HEIGHT,FRAME_WIDTH,3],dtype=tf.float32)
real_in = tf.compat.v1.placeholder(shape=[None,FRAME_HEIGHT,FRAME_WIDTH,3],dtype=tf.float32) #Real images

Gx_fill = generator(condition_in, 'fill') #Generates images from random z vectors
Dx_fill = discriminator(real_in, 'fill') #Produces probabilities for real images
Dg_fill = discriminator(Gx_fill, 'fill',reuse=True) #Produces probabilities for generator images

Gx_enh = generator(condition_in, 'enh') #Generates images from random z vectors
Dx_enh = discriminator(real_in, 'enh') #Produces probabilities for real images
Dg_enh = discriminator(Gx_enh, 'enh',reuse=True) #Produces probabilities for generator images

#These functions together define the optimization objective of the GAN.
d_loss_fill = -tf.reduce_mean(tf.math.log(Dx_fill) + tf.math.log(1.-Dg_fill)) #This optimizes the discriminator.
#For generator we use traditional GAN objective as well as L1 loss
g_loss_fill = -tf.reduce_mean(tf.math.log(Dg_fill)) + 100*tf.reduce_mean(tf.abs(Gx_fill - real_in)) #This optimizes the generator.

#These functions together define the optimization objective of the GAN.
d_loss_enh = -tf.reduce_mean(tf.math.log(Dx_enh) + tf.math.log(1.-Dg_enh)) #This optimizes the discriminator.
#For generator we use traditional GAN objective as well as L1 loss
g_loss_enh = -tf.reduce_mean(tf.math.log(Dg_enh)) + 100*tf.reduce_mean(tf.abs(Gx_enh - real_in)) #This optimizes the generator.

#The below code is responsible for applying gradient descent to update the GAN.
trainerD_fill = tf.compat.v1.train.AdamOptimizer(learning_rate=0.000002,beta1=0.5)
trainerG_fill = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00002,beta1=0.5)

#The below code is responsible for applying gradient descent to update the GAN.
trainerD_enh = tf.compat.v1.train.AdamOptimizer(learning_rate=0.000002,beta1=0.5)
trainerG_enh = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00002,beta1=0.5)

d_grads_fill = trainerD_fill.compute_gradients(d_loss_fill, slim.get_variables(scope='discriminator_fill'))
g_grads_fill = trainerG_fill.compute_gradients(g_loss_fill, slim.get_variables(scope='generator_fill'))

d_grads_enh = trainerD_enh.compute_gradients(d_loss_enh, slim.get_variables(scope='discriminator_enh'))
g_grads_enh = trainerG_enh.compute_gradients(g_loss_enh, slim.get_variables(scope='generator_enh'))

update_D_fill = trainerD_fill.apply_gradients(d_grads_fill)
update_G_fill = trainerG_fill.apply_gradients(g_grads_fill)

update_D_enh = trainerD_enh.apply_gradients(d_grads_enh)
update_G_enh = trainerG_enh.apply_gradients(g_grads_enh)

################################################################################
# TRAIN MODEL
################################################################################

if (mode == "train"):
    iterations = ceil((stop_frame - current_frame) / FRAME_GROUP) #Total number of iterations to use.
    sample_frequency = 200 #How often to generate sample gif of translated images.
    save_frequency = 2000 #How often to save model.

    init_fill = tf.compat.v1.global_variables_initializer()
    saver_fill = tf.compat.v1.train.Saver()
    init_enh = tf.compat.v1.global_variables_initializer()
    saver_enh = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess_fill, tf.compat.v1.Session() as sess_enh:
        sess_fill.run(init_fill)
        sess_enh.run(init_enh)
        if load_model == True:
            ckpt_fill = tf.train.get_checkpoint_state(ckpt_path)
            saver_fill.restore(sess_fill,ckpt_fill.model_checkpoint_path)
            ckpt_enh = tf.train.get_checkpoint_state(ckpt_path)
            saver_enh.restore(sess_enh,ckpt_enh.model_checkpoint_path)

        for i in range(iterations):
            # Load frame group
            current_frame, imagesX, imagesY =  get_frame_group(video, current_frame, start_frame, stop_frame, mode, random_frame)

            # Normalize frames
            ys = ((imagesY / 255.0) - 0.5) * 2.0
            xs = ((imagesX / 255.0) - 0.5) * 2.0

            assert not np.any(np.isnan(xs))
            assert not np.any(np.isnan(ys))

            for _ in range(EPOCHS):
                _,dLoss_fill = sess_fill.run([update_D_fill,d_loss_fill],feed_dict={real_in:ys,condition_in:xs}) #Update the discriminator
                _,gLoss_fill = sess_fill.run([update_G_fill,g_loss_fill],feed_dict={real_in:ys,condition_in:xs}) #Update the generator
                sample_G_fill = sess_fill.run(Gx_fill,feed_dict={condition_in:xs}) #Use new z to get sample images from generator.
                _,dLoss_enh = sess_enh.run([update_D_enh,d_loss_enh],feed_dict={real_in:ys,condition_in:sample_G_fill}) #Update the discriminator
                _,gLoss_enh = sess_enh.run([update_G_enh,g_loss_enh],feed_dict={real_in:ys,condition_in:sample_G_fill}) #Update the generator

            assert not (np.isnan(gLoss_fill) or np.isnan(dLoss_fill))
            assert not (np.isnan(gLoss_enh) or np.isnan(dLoss_enh))

            if i % sample_frequency == 0:
                frame_idx = np.random.randint(0,len(imagesX))
                print(f"Iteration: {i} Current Frame: {current_frame-FRAME_GROUP+frame_idx} GFill Loss: {str(gLoss_fill)} DFill Loss: {str(dLoss_fill)} GEnh Loss: {str(gLoss_enh)} DEnh Loss: {str(dLoss_enh)}")
                xs = ((np.reshape(imagesX[frame_idx],[1,FRAME_HEIGHT,FRAME_WIDTH,3]) / 255.0) - 0.5) * 2.0
                ys = ((np.reshape(imagesY[frame_idx],[1,FRAME_HEIGHT,FRAME_WIDTH,3]) / 255.0) - 0.5) * 2.0
                sample_G_fill = sess_fill.run(Gx_fill,feed_dict={condition_in:xs}) #Use new z to get sample images from generator.
                sample_G_enh = sess_enh.run(Gx_enh,feed_dict={condition_in:sample_G_fill}) #Use new z to get sample images from generator.

                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                #Save sample generator images for viewing training progress.
                time_str = str(datetime.now().strftime("%m%d%Y_%H_%M_%S"))

                width_diff = FRAME_WIDTH - ((FRAME_WIDTH // W_RATIO) * I_RATIO)
                width_start = width_diff // 2
                width_end = FRAME_WIDTH - (width_diff // 2)

                sample_frame = (np.around(((sample_G_enh[0] / 2.0) + 0.5) * 255.0)).astype(np.uint8)
                sample_frame_overlay = np.copy(sample_frame)
                sample_frame_overlay[:, width_start:width_end, :] = imagesX[frame_idx].astype(np.uint8)[:, width_start:width_end, :]

                Image.fromarray(imagesX[frame_idx][:,:,::-1].astype(np.uint8), 'RGB').save(os.path.join(output_path, str(time_str + "_input.png")))
                Image.fromarray(sample_frame[:,:,::-1], 'RGB').save(os.path.join(output_path, str(time_str + "_output.png")))
                Image.fromarray(sample_frame_overlay[:,:,::-1], 'RGB').save(os.path.join(output_path, str(time_str + "_output_overlay.png")))
                Image.fromarray(imagesY[frame_idx][:,:,::-1].astype(np.uint8), 'RGB').save(os.path.join(output_path, str(time_str + "_ref.png")))

            if ((i % save_frequency == 0) and (i != 0)):
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
                saver_fill.save(sess_fill,ckpt_path+'/widescreenify_fill_model_'+str(datetime.now().strftime("%m%d%Y_%H_%M_%S"))+'.ckpt')
                saver_enh.save(sess_enh,ckpt_path+'/widescreenify_enh_model_'+str(datetime.now().strftime("%m%d%Y_%H_%M_%S"))+'.ckpt')
                print("Saved Model")

        # Save of final model
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        saver_fill.save(sess_fill,ckpt_path+'/widescreenify_fill_model_'+str(datetime.now().strftime("%m%d%Y_%H_%M_%S"))+'.ckpt')
        saver_enh.save(sess_enh,ckpt_path+'/widescreenify_enh_model_'+str(datetime.now().strftime("%m%d%Y_%H_%M_%S"))+'.ckpt')
        print("Saved Model")

################################################################################
# PREDICT
################################################################################

if (mode == "test"):
    iterations = ceil((stop_frame - current_frame) / FRAME_GROUP) #Total number of iterations to use.

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    init_fill = tf.compat.v1.global_variables_initializer()
    saver_fill = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess_fill, tf.compat.v1.Session() as sess_enh:
        sess_fill.run(init_fill)
        sess_enh.run(init_enh)
        if load_model == True:
            ckpt_fill = tf.train.get_checkpoint_state(ckpt_path)
            saver_fill.restore(sess_fill,ckpt_fill.model_checkpoint_path)
            ckpt_enh = tf.train.get_checkpoint_state(ckpt_path)
            saver_enh.restore(sess_enh,ckpt_enh.model_checkpoint_path)

            output_file_name = ntpath.basename(video_path).split(".")[-2] + "_WIDE." + ntpath.basename(video_path).split(".")[-1]
            wide_video = cv2.VideoWriter(os.path.join(output_path, output_file_name), cv2.VideoWriter_fourcc('M','P','E','G'), frame_rate, (FRAME_WIDTH, FRAME_HEIGHT))

            for i in range(iterations):
                # Load frame group
                current_frame, imagesX, imagesY =  get_frame_group(video, current_frame, start_frame, stop_frame, mode, False)
                xs = ((imagesX / 255.0) - 0.5) * 2.0
                sample_G_fill = sess_fill.run(Gx_fill,feed_dict={condition_in:xs}) #Use new z to get sample images from generator.
                sample_G_enh = sess_enh.run(Gx_enh,feed_dict={condition_in:sample_G_fill}) #Use new z to get sample images from generator.

                output_frames = ((sample_G_enh / 2.0) + 0.5) * 255.0
                output_frames = np.around(output_frames).astype(np.uint8)

                for i, frame in enumerate(output_frames):
                    width_diff = FRAME_WIDTH - ((FRAME_WIDTH // W_RATIO) * I_RATIO)
                    width_start = width_diff // 2
                    width_end = FRAME_WIDTH - (width_diff // 2)

                    frame[:, width_start:width_end, :] = imagesY[i].astype(np.uint8)[:, width_start:width_end, :]

                    wide_video.write(frame)

            wide_video.release()

video.release()
