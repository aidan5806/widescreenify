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

import numpy as np
from math import ceil

import tensorflow as tf
import tf_slim as slim

from PIL import Image
import cv2

# PARAMS
FRAME_GROUP = 1
FRAME_HEIGHT = 144 # 720
FRAME_WIDTH = 256 # 1280
FRAME_RESCALE = 1
EPOCHS = 1
MODEL_SCALE = 32 # 8

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
def get_frame_group(video, current_frame, stop_frame, mode):
    group_stop_frame = current_frame + FRAME_GROUP if ((current_frame + FRAME_GROUP) <= stop_frame) else stop_frame
    group_frame_count = group_stop_frame - current_frame

    width_diff = FRAME_WIDTH - ((FRAME_WIDTH // W_RATIO) * I_RATIO)
    width_start = width_diff // 2
    width_end = FRAME_WIDTH - (width_diff // 2)

    input_frames = np.empty((group_frame_count, FRAME_HEIGHT, FRAME_WIDTH, 3), np.dtype('float32'))
    if (mode == "train"):
        output_frames = np.empty((group_frame_count, FRAME_HEIGHT, FRAME_WIDTH, 3), np.dtype('float32'))

    video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    for frame_index in range(0, group_frame_count):
        ret, frame = video.read()

        if not ret:
            print("Video read error")
            exit(0)

        if (FRAME_RESCALE):
            frame = cv2.resize(frame, dsize=(FRAME_WIDTH,FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC)

        if (mode == "train"):
            input_frames[frame_index, :, width_start:width_end, :] = frame[:, width_start:width_end, :]
            output_frames[frame_index] = frame
        else:
            input_frames[frame_index, :, width_start:width_end, :] = frame

    if (mode == "train"):
        return group_stop_frame, input_frames, output_frames
    else:
        return group_stop_frame, input_frames, None

#This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.compat.v1.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def generator(c):
    with tf.compat.v1.variable_scope('generator'):
        #Encoder
        enc0 = slim.conv2d(c,MODEL_SCALE*2,[3,3],padding="SAME",
            biases_initializer=None,activation_fn=lrelu,
            weights_initializer=initializer)
        enc0 = tf.nn.space_to_depth(enc0,2)

        enc1 = slim.conv2d(enc0,MODEL_SCALE*4,[3,3],padding="SAME",
            activation_fn=lrelu,normalizer_fn=slim.batch_norm,
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


def discriminator(bottom, reuse=False):
    with tf.compat.v1.variable_scope('discriminator'):
        filters = [MODEL_SCALE*1,MODEL_SCALE*2,MODEL_SCALE*4,MODEL_SCALE*4]

        #Programatically define layers
        for i in range(len(filters)):
            if i == 0:
                layer = slim.conv2d(bottom,filters[i],[3,3],padding="SAME",scope='d'+str(i),
                    biases_initializer=None,activation_fn=lrelu,stride=[3,3],
                    reuse=reuse,weights_initializer=initializer)
            else:
                layer = slim.conv2d(bottom,filters[i],[3,3],padding="SAME",scope='d'+str(i),
                    normalizer_fn=slim.batch_norm,activation_fn=lrelu,stride=[3,3],
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

args = parser.parse_args()

video_path = args.path
ckpt_path = args.ckpt
mode = args.mode
stop_frame = args.stop_frame
current_frame = args.start_frame
if (args.load_model or (mode == "test")):
    load_model = True
else:
    load_model = False
output_path = args.output

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

# current_frame, input_frames, output_frames = get_frame_group(video, current_frame, stop_frame, mode)

# print(input_frames.shape)
# print(output_frames.shape)
# print(current_frame)

# video.release()

# input_image = Image.fromarray(input_clip[FRAME_GROUP-1], 'RGB')
# target_image = Image.fromarray(target_clip[FRAME_GROUP-1], 'RGB')
# input_image.show()
# target_image.show()

# Normalize clips to range of 0 to 1
# input_clip = input_clip / 255.0
# target_clip = target_clip / 255.0

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#   except RuntimeError as e:
#     print(e)

# Define and connect model
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

#This initializaer is used to initialize all the weights of the network.
initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.02)

condition_in = tf.compat.v1.placeholder(shape=[None,FRAME_HEIGHT,FRAME_WIDTH,3],dtype=tf.float32)
real_in = tf.compat.v1.placeholder(shape=[None,FRAME_HEIGHT,FRAME_WIDTH,3],dtype=tf.float32) #Real images

Gx = generator(condition_in) #Generates images from random z vectors
Dx = discriminator(real_in) #Produces probabilities for real images
Dg = discriminator(Gx,reuse=True) #Produces probabilities for generator images

#These functions together define the optimization objective of the GAN.
d_loss = -tf.reduce_mean(tf.math.log(Dx) + tf.math.log(1.-Dg)) #This optimizes the discriminator.
#For generator we use traditional GAN objective as well as L1 loss
g_loss = -tf.reduce_mean(tf.math.log(Dg)) + 100*tf.reduce_mean(tf.abs(Gx - real_in)) #This optimizes the generator.

#The below code is responsible for applying gradient descent to update the GAN.
trainerD = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
trainerG = tf.compat.v1.train.AdamOptimizer(learning_rate=0.002,beta1=0.5)
d_grads = trainerD.compute_gradients(d_loss, slim.get_variables(scope='discriminator'))
g_grads = trainerG.compute_gradients(g_loss, slim.get_variables(scope='generator'))

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)

if (mode == "train"):
    iterations = ceil((stop_frame - current_frame) / FRAME_GROUP) #Total number of iterations to use.
    sample_frequency = 200 #How often to generate sample gif of translated images.
    save_frequency = 2000 #How often to save model.

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        if load_model == True:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver.restore(sess,ckpt.model_checkpoint_path)

        for i in range(iterations):
            # Load frame group
            current_frame, imagesX, imagesY =  get_frame_group(video, current_frame, stop_frame, mode)

            # Normalize frames
            ys = ((imagesY / 255.0) - 0.5) * 2.0
            xs = ((imagesX / 255.0) - 0.5) * 2.0

            assert not np.any(np.isnan(xs))
            assert not np.any(np.isnan(ys))

            for _ in range(EPOCHS):
                _,dLoss = sess.run([update_D,d_loss],feed_dict={real_in:ys,condition_in:xs}) #Update the discriminator
                _,gLoss = sess.run([update_G,g_loss],feed_dict={real_in:ys,condition_in:xs}) #Update the generator

            assert not (np.isnan(gLoss) or np.isnan(dLoss))

            if i % sample_frequency == 0:
                print("Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss))
                frame_idx = np.random.randint(0,len(imagesX))
                xs = ((np.reshape(imagesX[frame_idx],[1,FRAME_HEIGHT,FRAME_WIDTH,3]) / 255.0) - 0.5) * 2.0
                ys = ((np.reshape(imagesY[frame_idx],[1,FRAME_HEIGHT,FRAME_WIDTH,3]) / 255.0) - 0.5) * 2.0
                sample_G = sess.run(Gx,feed_dict={condition_in:xs}) #Use new z to get sample images from generator.

                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                #Save sample generator images for viewing training progress.
                time_str = str(datetime.now().strftime("%m%d%Y_%H_%M_%S"))

                width_diff = FRAME_WIDTH - ((FRAME_WIDTH // W_RATIO) * I_RATIO)
                width_start = width_diff // 2
                width_end = FRAME_WIDTH - (width_diff // 2)

                sample_frame = (np.around((sample_G[0] / 2.0) + 0.5) * 255.0).astype(np.uint8)
                sample_frame_overlay = np.copy(sample_frame)
                sample_frame_overlay[:, width_start:width_end, :] = imagesX[frame_idx].astype(np.uint8)[:, width_start:width_end, :]

                Image.fromarray(imagesX[frame_idx][:,:,::-1].astype(np.uint8), 'RGB').save(os.path.join(output_path, str(time_str + "_input.png")))
                Image.fromarray(sample_frame[:,:,::-1], 'RGB').save(os.path.join(output_path, str(time_str + "_output.png")))
                Image.fromarray(sample_frame_overlay[:,:,::-1], 'RGB').save(os.path.join(output_path, str(time_str + "_output_overlay.png")))
                Image.fromarray(imagesY[frame_idx][:,:,::-1].astype(np.uint8), 'RGB').save(os.path.join(output_path, str(time_str + "_ref.png")))

            if ((i % save_frequency == 0) and (i != 0)):
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
                saver.save(sess,ckpt_path+'/widescreenify_model_'+str(datetime.now().strftime("%m%d%Y_%H_%M_%S"))+'.ckpt')
                print("Saved Model")

        # Save of final model
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        saver.save(sess,ckpt_path+'/widescreenify_model_'+str(datetime.now().strftime("%m%d%Y_%H_%M_%S"))+'.ckpt')
        print("Saved Model")

if (mode == "test"):
    iterations = ceil((stop_frame - current_frame) / FRAME_GROUP) #Total number of iterations to use.

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        if load_model == True:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver.restore(sess,ckpt.model_checkpoint_path)

            output_file_name = ntpath.basename(video_path).split(".")[-2] + "_WIDE." + ntpath.basename(video_path).split(".")[-1]
            wide_video = cv2.VideoWriter(os.path.join(output_path, output_file_name), cv2.VideoWriter_fourcc('M','P','E','G'), frame_rate, (FRAME_WIDTH, FRAME_HEIGHT))

            for i in range(iterations):
                # Load frame group
                current_frame, imagesX, imagesY =  get_frame_group(video, current_frame, stop_frame, mode)
                xs = ((imagesX / 255.0) - 0.5) * 2.0
                sample_G = sess.run(Gx,feed_dict={condition_in:xs}) #Use new z to get sample images from generator.

                output_frames = ((sample_G / 2.0) + 0.5) * 255.0
                output_frames = np.around(output_frames).astype(np.uint8)

                for i, frame in enumerate(output_frames):
                    width_diff = FRAME_WIDTH - ((FRAME_WIDTH // W_RATIO) * I_RATIO)
                    width_start = width_diff // 2
                    width_end = FRAME_WIDTH - (width_diff // 2)

                    frame[:, width_start:width_end, :] = imagesX[i].astype(np.uint8)[:, width_start:width_end, :]

                    wide_video.write(frame)

            wide_video.release()

video.release()

# https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/