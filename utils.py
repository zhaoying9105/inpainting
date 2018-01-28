import os
import math
import numpy as np
import tensorflow as tf
from PIL import Image
import itertools
from glob import glob
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from skimage.io import imread

def save(sess, saver, checkpoint_dir, step):
  model_name = "model"
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def load(sess, saver, checkpoint_dir):
  import re
  print(' [*] Reading checkpoints...')

  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    counter = int(next(re.finditer("([0-9]+)(?!.*[0-9])", ckpt_name)).group(0))
    print(" [*] Success to read {}".format(ckpt_name))
    return True, counter
  else:
    print(' [*] Failed to find a checkpoint')
    return False, 0

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]
def dataset_files(root):
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))

def read_batch_image(file_list, batch_size):
  lenght = len(file_list)
  print('lenght: ',lenght)
  indices = np.arange(lenght)
  np.random.shuffle(indices)
  for i in range(lenght // batch_size):
    images = []
    for index in range(i*batch_size,(i+1)*batch_size):
      image = imread(file_list[index],mode='RGB').astype(np.float)
      if image.shape != (64,64,3):
        print(file_list[i])
      else:
        images.append(image)
    yield np.array(images)

def read_by_batch(file_object, batch_size, data_shape, label=False):
  """
    read file one batch at a time, data shape shoud be HWC,
    in NinaPro dataset, data_shape[0] is the frame numbers,
    and data_shape[1] is the size of each frame of ninapro.
  """
  assert len(data_shape) == 3, 'Wrong data_shape: ' + str(data_shape)

  while True:
    # size equals data size plus label size
    data_size = data_shape[0] * data_shape[1] * data_shape[2]
    if label:
      data_batch = np.fromfile(file_object, dtype=np.uint8,
                               count=(data_size + 1) * batch_size)
    else:
      data_batch = np.fromfile(file_object, dtype=np.uint8,
                               count=data_size * batch_size)
    if data_batch is None:
      break
    if label:
      data_batch = np.reshape(data_batch, (-1, data_size + 1))
      images = np.reshape(
          data_batch[:, :data_size], (-1, data_shape[0], data_shape[1], data_shape[2]))
      labels = data_batch[:, -1]
      yield images, labels
    else:
      images = np.reshape(
          data_batch, (-1, data_shape[0], data_shape[1], data_shape[2]))
      yield images


"""
def preprocess_image(images, batch_size, image_size,
                      image_dim):  # single central blocks
"""
def preprocess_image(images, batch_size, image_size, 
                     image_dim,random_block=True, mask_percent=0.25):  # single random blocks

  if random_block:
    masks = np.random.choice([0,1],size=(batch_size, image_size, image_size,
                   image_dim),p=[mask_percent,1-mask_percent])
  else:
    masks = np.ones((batch_size, image_size, image_size,
                   image_dim), dtype=np.float32)
    masks[:, int(image_size * 0.25):int(image_size * 0.75),
          int(image_size * 0.25):int(image_size * 0.75), :] = 0
  masked_images = np.multiply(images, masks)
  print('masks shape: ',masks.shape)
  print('images shape: ',images.shape)
  return masked_images,images,masks


def combine_images(images):
  num = images.shape[0]
  width = int(math.sqrt(num))
  height = int(math.ceil(float(num) / width))
  shape = images.shape[1:4]
  output_image = np.zeros(
      (height * shape[0], width * shape[1], shape[2]), dtype=images.dtype)
  for index, img in enumerate(images):
    i = int(index / width)
    j = index % width
    output_image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = \
        img[:, :, :]
  return output_image


def save_images(images, epoch, index, sample_dir):
  shape = images.shape
  image = combine_images(images)
  image = image * 127.5 + 127.5
  image_path = os.path.join(sample_dir, str(epoch) + '_' + str(index) + '.jpg')
  if shape[3] == 1:
    image = np.squeeze(image)
    Image.fromarray(image.astype(np.uint8), mode='L').save(image_path)
  else:
    Image.fromarray(image.astype(np.uint8)).save(image_path)


def compute_psnr_ssim(images1, images2):
  images1 = images1 * 127.5 + 127.5
  images2 = images2 * 127.5 + 127.5
  images1 = images1.astype(np.uint8)
  images2 = images2.astype(np.uint8)
  batch_size = np.shape(images1)[0]
  psnr = np.zeros((batch_size))
  ssim = np.zeros((batch_size))
  for idx in range(batch_size):
    psnr[idx] = compare_psnr(images1[idx], images2[idx])
    ssim[idx] = compare_ssim(images1[idx], images2[idx], multichannel=True)

  return np.mean(psnr), np.mean(ssim)


def extend_array_by_index(inputs, index, full_height, full_width=None):
  """
  inputs: shape of NHWC
  index: shape of (N, 2)
  full_height & full_width: the height and width after extend,
          if full_width is None, then full_width = full_height
  """
  shape = inputs.get_shape().as_list()
  if index.get_shape().as_list()[0] != shape[0]:
    raise 'Inputs tensor shape[0] does not match index shape[0]'

  batch_size = shape[0]
  height = shape[1]
  width = shape[2]
  channel = shape[3]

  if full_width is None:
    full_width = full_height

  indices = None
  for idx in range(batch_size):
    idx_start1 = tf.cast(index[idx, 0], tf.int32)
    idx_end1 = tf.cast(index[idx, 0] + height, tf.int32)
    idx_start2 = tf.cast(index[idx, 1], tf.int32)
    idx_end2 = tf.cast(index[idx, 1] + width, tf.int32)

    indice_0 = tf.ones([height * width, 1], dtype=tf.int32) * idx

    indice_1 = tf.range(idx_start1, idx_end1)
    indice_1 = tf.reshape(indice_1, [-1, 1])
    indice_1 = tf.tile(indice_1, [1, width])
    indice_1 = tf.reshape(indice_1, [-1, 1])

    indice_2 = tf.range(idx_start2, idx_end2)
    indice_2 = tf.reshape(indice_2, [-1, 1])
    indice_2 = tf.tile(indice_2, [height, 1])

    indice = tf.concat([indice_0, indice_1, indice_2], axis=1)
    if indices is None:
      indices = indice
    else:
      indices = tf.concat([indices, indice], axis=0)

  values = tf.reshape(inputs, [-1, channel])
  scatter = tf.scatter_nd(indices, values, tf.constant([
      batch_size, full_height, full_width, channel]))
  return scatter
