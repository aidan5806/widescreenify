################################################################################
# widescreenify.py
# Aidan Maycock
# Heavily based on a Pix2Pix implementation by Arthur Juliani
# https://github.com/awjuliani/Pix2Pix-Film/blob/master/Pix2Pix.ipynb
################################################################################

import argparse
import os

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

from PIL import Image
import cv2

# PARAMS
FRAME_GROUP = 100
FRAME_HEIGHT = 720
FRAME_WIDTH = 1280

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

    input_frames = np.empty((group_frame_count, FRAME_HEIGHT, FRAME_WIDTH, 3), np.dtype('uint8'))
    if (mode == "train"):
        output_frames = np.empty((group_frame_count, FRAME_HEIGHT, FRAME_WIDTH, 3), np.dtype('uint8'))

    video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    for frame_index in range(0, group_frame_count):
        ret, frame = video.read()

        if not ret:
            print("Video read error")
            exit(0)

        input_frames[frame_index, :, width_start:width_end, :] = frame[:, width_start:width_end, :]
        if (mode == "train"):
            output_frames[frame_index] = frame

    if (mode == "train"):
        return group_stop_frame, input_frames, output_frames
    else:
        return group_stop_frame, input_frames, None


def generator(c):
    with tf.variable_scope('generator'):
        #Encoder
        enc0 = slim.conv2d(c,64,[3,3],padding="SAME",
            biases_initializer=None,activation_fn=lrelu,
            weights_initializer=initializer)
        enc0 = tf.space_to_depth(enc0,2)

        enc1 = slim.conv2d(enc0,128,[3,3],padding="SAME",
            activation_fn=lrelu,normalizer_fn=slim.batch_norm,
            weights_initializer=initializer)
        enc1 = tf.space_to_depth(enc1,2)

        enc2 = slim.conv2d(enc1,128,[3,3],padding="SAME",
            normalizer_fn=slim.batch_norm,activation_fn=lrelu,
            weights_initializer=initializer)
        enc2 = tf.space_to_depth(enc2,2)

        enc3 = slim.conv2d(enc2,256,[3,3],padding="SAME",
            normalizer_fn=slim.batch_norm,activation_fn=lrelu,
            weights_initializer=initializer)
        enc3 = tf.space_to_depth(enc3,2)

        #Decoder
        gen0 = slim.conv2d(
            enc3,num_outputs=256,kernel_size=[3,3],
            padding="SAME",normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.elu, weights_initializer=initializer)
        gen0 = tf.depth_to_space(gen0,2)

        gen1 = slim.conv2d(
            tf.concat([gen0,enc2],3),num_outputs=256,kernel_size=[3,3],
            padding="SAME",normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.elu,weights_initializer=initializer)
        gen1 = tf.depth_to_space(gen1,2)

        gen2 = slim.conv2d(
            tf.concat([gen1,enc1],3),num_outputs=128,kernel_size=[3,3],
            padding="SAME",normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.elu,weights_initializer=initializer)
        gen2 = tf.depth_to_space(gen2,2)

        gen3 = slim.conv2d(
            tf.concat([gen2,enc0],3),num_outputs=128,kernel_size=[3,3],
            padding="SAME",normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.elu, weights_initializer=initializer)
        gen3 = tf.depth_to_space(gen3,2)

        g_out = slim.conv2d(
            gen3,num_outputs=3,kernel_size=[1,1],padding="SAME",
            biases_initializer=None,activation_fn=tf.nn.tanh,
            weights_initializer=initializer)
        return g_out


def discriminator(bottom, reuse=False):
    with tf.variable_scope('discriminator'):
        filters = [32,64,128,128]

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

        dis_full = slim.fully_connected(slim.flatten(bottom),1024,activation_fn=lrelu,scope='dl',
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
parser.add_argument("-m", "--mode", type=str, default="train", help="Set to either train or test mode.")
parser.add_argument("-c", "--ckpt", type=str, default="test.ckpt", help="Set path to model checkpoint file.")
parser.add_argument("-n", "--new_ckpt", type=str, help="Set path to new model checkpoint file, to avoid overwriting previous model. If not included imported checkpoint will be overwritten.")
parser.add_argument("-s", "--start_frame", type=int, default=0, help="Set the starting frame for video ingest.")
parser.add_argument("-f", "--stop_frame", type=int, help="Set maximum number of frames to train or test by indicating the stop frame.")

args = parser.parse_args()

video_path = args.path
ckpt_path = args.ckpt
new_ckpt_path = args.new_ckpt
mode = args.mode
stop_frame = args.stop_frame
current_frame = args.start_frame

# Ingest video clip for training
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

if not check_props(frame_width, frame_height, mode):
    print(f"Incompatible input video dimensions:: {frame_width}x{frame_height}")
    exit(1)

current_frame, input_frames, output_frames = get_frame_group(video, current_frame, stop_frame, mode)

print(input_frames.shape)
print(output_frames.shape)
print(current_frame)

video.release()

# input_image = Image.fromarray(input_clip[FRAME_GROUP-1], 'RGB')
# target_image = Image.fromarray(target_clip[FRAME_GROUP-1], 'RGB')
# input_image.show()
# target_image.show()


# Normalize clips to range of 0 to 1
# input_clip = input_clip / 255.0
# target_clip = target_clip / 255.0

# # Define and connect model
# tf.reset_default_graph()

# #This initializaer is used to initialize all the weights of the network.
# initializer = tf.truncated_normal_initializer(stddev=0.02)

# condition_in = tf.placeholder(shape=[None,FRAME_HEIGHT,FRAME_WIDTH,3],dtype=tf.float32)
# real_in = tf.placeholder(shape=[None,FRAME_HEIGHT,FRAME_WIDTH,3],dtype=tf.float32) #Real images

# Gx = generator(condition_in) #Generates images from random z vectors
# Dx = discriminator(real_in) #Produces probabilities for real images
# Dg = discriminator(Gx,reuse=True) #Produces probabilities for generator images

# #These functions together define the optimization objective of the GAN.
# d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
# #For generator we use traditional GAN objective as well as L1 loss
# g_loss = -tf.reduce_mean(tf.log(Dg)) + 100*tf.reduce_mean(tf.abs(Gx - real_in)) #This optimizes the generator.

# #The below code is responsible for applying gradient descent to update the GAN.
# trainerD = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
# trainerG = tf.train.AdamOptimizer(learning_rate=0.002,beta1=0.5)
# d_grads = trainerD.compute_gradients(d_loss,slim.get_variables(scope='discriminator'))
# g_grads = trainerG.compute_gradients(g_loss, slim.get_variables(scope='generator'))

# update_D = trainerD.apply_gradients(d_grads)
# update_G = trainerG.apply_gradients(g_grads)

# batch_size = 4 #Size of image batch to apply at each iteration.
# iterations = 500000 #Total number of iterations to use.
# subset_size = 5000 #How many images to load at a time, will vary depending on available resources
# frame_directory = './frames' #Directory where training images are located
# sample_directory = './samples' #Directory to save sample images from generator in.
# model_directory = './model' #Directory to save trained model to.
# sample_frequency = 200 #How often to generate sample gif of translated images.
# save_frequency = 5000 #How often to save model.
# load_model = False #Whether to load the model or begin training from scratch.

# subset = 0
# dataS = sorted(glob(os.path.join(frame_directory, "*.png")))
# total_subsets = len(dataS)/subset_size
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#     if load_model == True:
#         ckpt = tf.train.get_checkpoint_state(model_directory)
#         saver.restore(sess,ckpt.model_checkpoint_path)

#     imagesY,imagesX = loadImages(dataS[0:subset_size],False, np.random.randint(0,2)) #Load a subset of images
#     print "Loaded subset " + str(subset)
#     draw = range(len(imagesX))
#     for i in range(iterations):
#         if i % (subset_size/batch_size) != 0 or i == 0:
#             batch_index = np.random.choice(draw,size=batch_size,replace=False)
#         else:
#             subset = np.random.randint(0,total_subsets+1)
#             imagesY,imagesX = loadImages(dataS[subset*subset_size:(subset+1)*subset_size],False, np.random.randint(0,2))
#             print "Loaded subset " + str(subset)
#             draw = range(len(imagesX))
#             batch_index = np.random.choice(draw,size=batch_size,replace=False)

#         ys = (np.reshape(imagesY[batch_index],[batch_size,height,width,3]) - 0.5) * 2.0 #Transform to be between -1 and 1
#         xs = (np.reshape(imagesX[batch_index],[batch_size,height,width,3]) - 0.5) * 2.0
#         _,dLoss = sess.run([update_D,d_loss],feed_dict={real_in:ys,condition_in:xs}) #Update the discriminator
#         _,gLoss = sess.run([update_G,g_loss],feed_dict={real_in:ys,condition_in:xs}) #Update the generator
#         if i % sample_frequency == 0:
#             print "Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss)
#             start_point = np.random.randint(0,len(imagesX)-32)
#             xs = (np.reshape(imagesX[start_point:start_point+32],[32,height,width,3]) - 0.5) * 2.0
#             ys = (np.reshape(imagesY[start_point:start_point+32],[32,height,width,3]) - 0.5) * 2.0
#             sample_G = sess.run(Gx,feed_dict={condition_in:xs}) #Use new z to get sample images from generator.
#             allS = np.concatenate([xs,sample_G,ys],axis=1)
#             if not os.path.exists(sample_directory):
#                 os.makedirs(sample_directory)
#             #Save sample generator images for viewing training progress.
#             make_gif(allS,'./'+sample_directory+'/a_vid'+str(i)+'.gif',
#                 duration=len(allS)*0.2,true_image=False)
#         if i % save_frequency == 0 and i != 0:
#             if not os.path.exists(model_directory):
#                 os.makedirs(model_directory)
#             saver.save(sess,model_directory+'/model-'+str(i)+'.cptk')
#             print "Saved Model"

# test_directory = './test_frames' #Directory to load test frames from
# subset_size = 5000
# batch_size = 60 # Size of image batch to apply at each iteration. Will depend of available resources.
# sample_directory = './test_samples' #Directory to save sample images from generator in.
# model_directory = './model' #Directory to save trained model to.
# load_model = True #Whether to load a saved model.

# dataS = sorted(glob(os.path.join(test_directory, "*.png")))
# subset = 0
# total_subsets = len(dataS)/subset_size
# iterations = subset_size / batch_size #Total number of iterations to use.


# if not os.path.exists(sample_directory):
#     os.makedirs(sample_directory)

# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#     if load_model == True:
#         ckpt = tf.train.get_checkpoint_state(model_directory)
#         saver.restore(sess,ckpt.model_checkpoint_path)
#         for s in range(total_subsets):
#             generated_frames = []
#             _,imagesX = loadImages(dataS[s*subset_size:s*subset_size+subset_size],False, False) #Load a subset of images
#             for i in range(iterations):
#                 start_point = i * batch_size
#                 xs = (np.reshape(imagesX[start_point:start_point+batch_size],[batch_size,height,width,3]) - 0.5) * 2.0
#                 sample_G = sess.run(Gx,feed_dict={condition_in:xs}) #Use new z to get sample images from generator.
#                 #allS = np.concatenate([xs,sample_G],axis=2)
#                 generated_frames.append(sample_G)
#             generated_frames = np.vstack(generated_frames)
#             for i in range(len(generated_frames)):
#                 im = Image.fromarray(((generated_frames[i]/2.0 + 0.5) * 256).astype('uint8'))
#                 im.save('./'+sample_directory+'/frame'+str(s*subset_size + i)+'.png')
#             #make_gif(generated_frames,'./'+sample_directory+'/a_vid'+str(i)+'.gif',
#             #    duration=len(generated_frames)/10.0,true_image=False)