import json
import math
import os
import collections
import os.path

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from PIL import Image


def combine_dicts(name_to_dict):
  combined = {}
  for dict_name, current_dict in name_to_dict.iteritems():
    for name, val in current_dict.iteritems():
      combined[dict_name + '_' + name] = val
  return combined

def mean_squared_error(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
  if labels is None:
    raise ValueError("labels must not be None.")
  if predictions is None:
    raise ValueError("predictions must not be None.")
  with tf.name_scope(scope, "mean_squared_error",
                      (predictions, labels, weights)) as scope:
    # This will cast float16 to float32 which is not what we want.
    # predictions = math_ops.to_float(predictions)
    # labels = math_ops.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    losses = tf.squared_difference(predictions, labels)
    return tf.losses.compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)

# Taken from https://github.com/tensorflow/tensorflow/issues/8246 by qianyizhang.
def tf_repeat(tensor, repeats):
  """
  Args:

  input: A Tensor. 1-D or higher.
  repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

  Returns:

  A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
  """
  if isinstance(repeats, tuple):
    repeats = list(repeats)
  assert len(repeats) == len(tensor.shape), 'repeat length must be the same as the number of dimensions in input.'
  with tf.variable_scope("repeat"):
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
    new_shape = tf.TensorShape([tensor.shape[i] * repeats[i] for i in range(len(repeats))])
    repeated_tensor = tf.reshape(tiled_tensor,new_shape)
  return repeated_tensor

def grayscale_to_heatmap(gray, is_bgr=False):
  four = tf.constant(4, dtype=gray.dtype)
  zero = tf.constant(0, dtype=gray.dtype)
  one = tf.constant(1, dtype=gray.dtype)

  r = tf.clip_by_value(tf.minimum(four * gray - tf.constant(1.5, dtype=gray.dtype),
                                  -four * gray + tf.constant(4.5, dtype=gray.dtype)), zero, one)
  g = tf.clip_by_value(tf.minimum(four * gray - tf.constant(0.5, dtype=gray.dtype),
                                  -four * gray + tf.constant(3.5, dtype=gray.dtype)), zero, one)
  b = tf.clip_by_value(tf.minimum(four * gray + tf.constant(0.5, dtype=gray.dtype),
                                  -four * gray + tf.constant(2.5, dtype=gray.dtype)), zero, one)
  if is_bgr:
    return tf.concat((b, g, r), axis=-1)
  else:
    return tf.concat((r, g, b), axis=-1)



def get_latest_checkpoint_path(path):
  assert(path)
  if tf.gfile.IsDirectory(path):
    checkpoint_path = tf.train.latest_checkpoint(path)
  else:
    checkpoint_path = path
  return checkpoint_path


def fp16_friendly_leaky_relu(features, alpha=0.2, name=None):
  """Compute the Leaky ReLU activation function.

  "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
  http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf

  Args:
    features: A `Tensor` representing preactivation values.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).

  Returns:
    The activation value.
  """
  with tf.name_scope(name, "LeakyRelu", [features, alpha]):
    features = tf.convert_to_tensor(features, name="features", dtype=features.dtype)
    alpha = tf.convert_to_tensor(alpha, name="alpha", dtype=features.dtype)
    return tf.maximum(alpha * features, features)

def save_image(img, filename):
  """Saves a numpy image to `filename`."""
  # Unfortunately we cannot use _post_process_image() because the graph is already finalized.
  img = img * 255.0  # Because our post processed image is from 0~1.0
  img = img.astype(np.int32)
  if img.shape[-1] > 3:
    # Convert the image into one channel by summing all channels together
    img = np.sum(img, axis=-1, keepdims=True)
  img[img < 0] = 0
  img[img >= 256] = 255
  img = np.uint8(img)
  if img.shape[-1] == 1:
    img = np.squeeze(img, -1)
  img = Image.fromarray(img)
  img.save(filename)

def get_image_height(image):
  return int(image.shape[-3])

def get_image_width(image):
  return int(image.shape[-2])

def reshape_to_power2(image, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
  """Given a square image, returns a reshaped (smaller) image that has height and width equals to power of 2."""
  input_hw = get_image_height(image)
  assert input_hw == get_image_width(image), 'Currently `reshape_to_power2` only supports square images.'
  image_reshaped = image
  target_hw = int(math.log(input_hw, 2))
  if math.log(input_hw, 2) != int(math.log(input_hw, 2)):
    image_reshaped = tf.image.resize_images(target_hw, (target_hw, target_hw), method)
  return image_reshaped

def xy_to_one_hot(xy, image_size, do_transpose=True):
  # xy shape: [batch_size, 2 * num_landmarks].
  assert xy is not None
  xy_reshaped = tf.reshape(xy, tf.TensorShape([xy.shape[0], int(xy.shape[1]) / 2, 2]))
  xy_reshaped = tf.minimum(tf.cast(tf.round(xy_reshaped * image_size), tf.int32), image_size - 1)  # Prevent overflow.
  xy_reshaped = xy_reshaped[..., 0] + xy_reshaped[..., 1] * image_size  # format: x + y * width.
  landmark = tf.one_hot(xy_reshaped, image_size * image_size)
  if do_transpose:
    landmark = tf.transpose(landmark, (0, 2, 1))
    landmark = tf.reshape(landmark,
                          tf.TensorShape([xy_reshaped.shape[0], image_size, image_size, xy_reshaped.shape[1]]))
  return landmark

def get_xy_landmark_prediction_loss(xy, landmark_predictions, weights, loss_collections):
  xy_reshaped = tf.reshape(xy, tf.TensorShape([xy.shape[0], int(xy.shape[1]) / 2, 2]))
  width = int(landmark_predictions.shape[1])
  height = int(landmark_predictions.shape[2])
  assert width == height, 'Currently do not support non-squares.'
  width_per_block = 1.0 / width
  dx = tf.mod(xy_reshaped[..., 0], width_per_block) / width_per_block
  dx = tf_repeat(tf.expand_dims(tf.expand_dims(dx, 1), 2), [1, width, height, 1])

  height_per_block = 1.0 / height
  dy = tf.mod(xy_reshaped[..., 1], height_per_block) / height_per_block
  dy = tf_repeat(tf.expand_dims(tf.expand_dims(dy, 1), 2), [1, width, height, 1])
  c = xy_to_one_hot(xy, width, do_transpose=False)
  c_transposed = tf.transpose(c, (0, 2, 1))
  c_transposed = tf.reshape(c_transposed,
                            tf.TensorShape([xy_reshaped.shape[0], width, width, xy_reshaped.shape[1]]))
  landmark_confidence_reshaped = tf.transpose(landmark_predictions[..., 2], (0, 3, 1, 2))
  landmark_confidence_reshaped = tf.reshape(landmark_confidence_reshaped,
                                            tf.TensorShape([landmark_confidence_reshaped.shape[0],
                                                            landmark_confidence_reshaped.shape[1],
                                                            landmark_confidence_reshaped.shape[2] *
                                                            landmark_confidence_reshaped.shape[3],]))

  dx_loss = tf.losses.absolute_difference(dx * c_transposed, landmark_predictions[..., 0] * c_transposed, weights=weights,
                               scope='landmark_dx_loss', loss_collection=loss_collections[0])
  dy_loss = tf.losses.absolute_difference(dy * c_transposed, landmark_predictions[..., 1] * c_transposed, weights=weights,
                               scope='landmark_dy_loss', loss_collection=loss_collections[0])
  c_loss = tf.losses.softmax_cross_entropy(c, landmark_confidence_reshaped, weights=weights,
                                  scope='landmark_confidence_loss', loss_collection=loss_collections[0])
  for i in range(1, len(loss_collections)):
    for loss in [dx_loss, dy_loss, c_loss]:
      tf.losses.add_loss(loss, loss_collections[i])

def visualize_landmark_predictions(landmark_predictions, image_size, use_softmax=True):
  grid_size = landmark_predictions.shape[1]
  assert grid_size == landmark_predictions.shape[2]
  # Row = [[0, 1, 2, 3...], [0, 1, 2, 3...]...]. Column = [[0, 0, ..], [1, 1,...]...]
  row, column = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
  dtype = landmark_predictions.dtype
  row = tf.cast(row, dtype) / tf.cast(grid_size, dtype)
  column = tf.cast(column, dtype) / tf.cast(grid_size, dtype)
  rc = tf.stack((row, column), axis=-1)
  rc_reshaped = tf.expand_dims(tf.expand_dims(rc, 0), 3)
  rc_reshaped = tf_repeat(rc_reshaped,
                                    [landmark_predictions.shape[0], 1, 1, landmark_predictions.shape[3], 1])
  if True:
    landmark_confidence_reshaped = tf.transpose(landmark_predictions[..., 2], (0, 3, 1, 2))
    landmark_confidence_reshaped = tf.reshape(landmark_confidence_reshaped,
                                              tf.TensorShape([landmark_confidence_reshaped.shape[0],
                                                              landmark_confidence_reshaped.shape[1],
                                                              landmark_confidence_reshaped.shape[2] *
                                                              landmark_confidence_reshaped.shape[3], ]))
    c = layers.softmax(landmark_confidence_reshaped)
  else:
    c = tf.sigmoid(landmark_predictions[..., 2])
  c = tf.transpose(c, (0, 2, 1))
  # c = tf.reshape(c, tf.TensorShape(
  #   [c.shape[0], grid_size, grid_size, c.shape[2]]))  # [batch size, height, width, num_landmarks]
  xy = rc_reshaped + landmark_predictions[..., 0:2] / int(
    grid_size)  # [batch size, height, width, num_landmarks, 2]
  xy_reshaped = tf.reshape(xy, tf.TensorShape([xy.shape[0], xy.shape[1] * xy.shape[2] * xy.shape[3] * xy.shape[4]]))
  xy_one_hot = xy_to_one_hot(xy_reshaped, image_size)
  xy_one_hot_weighed = xy_one_hot * tf.expand_dims(tf.expand_dims(layers.flatten(c), 1), 2)
  xy_one_hot_weighed = tf.reduce_sum(xy_one_hot_weighed, axis=-1, keepdims=True)
  return xy_one_hot_weighed

def safe_print(string, *args):
  try:
    print string.format(*args)
  except:
    try:
      print unicode(string).format(*args)
    except:
      print('cannot print string.')


def process_mutually_exclusive_labels(labels, classification_threshold, labels_id_to_group=None):
  """Given a numpy array of labels and the groups each label belongs to, output the maximum values for each group
  and set the non-max to 0."""
  # TODO: hardcoded files for now.
  if labels_id_to_group is None:
    labels_id_to_group = get_tags_dict('./datasets/anime_face_tag_list.txt', 0, 2)

  ret = [0.0 for _ in range(len(labels))]
  group_vals = collections.defaultdict(list)
  for i, val in enumerate(labels):
    group = labels_id_to_group.get(i, None)
    if group is not None:
      group_vals[group].append((i, val))

  hair_color_missing = True
  eye_color_missing = True

  for group, vals in group_vals.iteritems():
    max_item = max(vals, key=lambda x: x[1])
    ret[max_item[0]] = max_item[1]
    # TODO: hard coded hair and eye color missing detection.
    if group == '2':
      if max_item[1] >= classification_threshold:
        hair_color_missing = False
    if group == '3':
      if max_item[1] >= classification_threshold:
        eye_color_missing = False

  # Do not output any label if eye color or hair color is missing.
  if eye_color_missing or hair_color_missing:
    return [0.0 for _ in range(len(labels))]
  else:
    return ret


def get_tags_dict(path, key_column_index, value_column_index):
  """Returns a list of ..."""
  ret = {}
  with open(path, 'r') as f:
    for i, line in enumerate(f):
      if len(line):
        whole_line = line.rstrip('\n')
        content = whole_line.split('\t')
        key = i if key_column_index is None else int(content[key_column_index])
        value = whole_line if value_column_index is None else content[value_column_index]
        ret[key] = value
  return ret

def get_no_ext_base(file_name):
  return os.path.splitext(os.path.basename(file_name))[0]

def concat_cond_vector(layer, cond_vector):
  # Resize conditional layer to the same height and width as layer.
  assert len(cond_vector.shape) == 2
  resized_conditional_layer = tf.expand_dims(tf.expand_dims(cond_vector, axis=1), axis=2)
  resized_conditional_layer = tf_repeat(resized_conditional_layer, (1, layer.shape[1], layer.shape[2], 1))
  if resized_conditional_layer.dtype != cond_vector.dtype:
    resized_conditional_layer = tf.cast(resized_conditional_layer, cond_vector.dtype)
  return tf.concat((layer, resized_conditional_layer), axis=-1)


def get_landmark_dict(directories, landmark_file_name, do_join=True):
  ret = collections.defaultdict(list)
  if do_join:
    for directory in directories:
      landmark_file_path = os.path.join(directory, landmark_file_name)
      with open(landmark_file_path) as f:
        landmarks = collections.defaultdict(list)
        for line in f:
          landmark = json.loads(line)
          if 'file' in landmark:
            landmarks[os.path.basename(landmark['file'])].append(landmark)
      ret.update(landmarks)
  else:
    with open(landmark_file_name) as f:
      for line in f:
        landmark = json.loads(line)
        if 'file' in landmark:
          ret[os.path.basename(landmark['file'])].append(landmark)
  return ret


def get_relative_xywh(json_object, relative_to_x, relative_to_y, width, height):
  x, y, w, h= _get_xywh(json_object)
  relative_x = x - relative_to_x
  relative_y = y - relative_to_y
  if relative_x < 0 or relative_y < 0:
    raise ValueError('relative_x < 0 or relative_y < 0: relative_x = %d relative_y = %d' %(relative_x, relative_y))
  if (relative_x + w) / float(width) >= 1 or (relative_y + h) / float(height) >= 1:
    raise ValueError('(relative_x + w) / float(width) >= 1 or (relative_y + h) / float(height) >= 1:'
                     ' (relative_x + w) / float(width) = %f (relative_y + h) / float(height) = %f'
                     %((relative_x + w) / float(width), (relative_y + h) / float(height)))
  return relative_x / float(width), relative_y / float(height), w / float(width), h / float(height)


def _get_xywh(json_object):
  return json_object['x'], json_object['y'], json_object['height'], json_object['width']


def expand_xywh(x, y, w, h, image_w, image_h, hw_expansion_rate):
  # Expand h, w on each side by `hw_expansion_rate`.
  x_expanded = max(0, x - int(w * hw_expansion_rate))
  y_expanded = max(0, y - int(h * hw_expansion_rate))
  x_end_expanded = min(image_w, x + int(w * (1.0 + hw_expansion_rate)))
  y_end_expanded = min(image_h, y + int(h * (1.0 + hw_expansion_rate)))

  return x_expanded, y_expanded, x_end_expanded - x_expanded, y_end_expanded- y_expanded,


def unevenly_expand_xywh(x, y, w, h, image_w, image_h, left_w_ratio, right_w_ratio, top_h_ratio, bottom_h_ratio):
  # Expand h, w on each side by `HW_EXPANSION_RATE`.
  x_expanded = max(0, x - int(w * left_w_ratio))
  y_expanded = max(0, y - int(h * top_h_ratio))
  x_end_expanded = min(image_w, x + int(w * (1.0 + right_w_ratio)))
  y_end_expanded = min(image_h, y + int(h * (1.0 + bottom_h_ratio)))

  return x_expanded, y_expanded, x_end_expanded - x_expanded, y_end_expanded- y_expanded,


def get_relative_xy(xy, relative_to_x, relative_to_y, width, height):
  x, y, = xy
  relative_x = x - relative_to_x
  relative_y = y - relative_to_y
  if relative_x < 0 or relative_y < 0:
    raise ValueError('relative_x < 0 or relative_y < 0: relative_x = %d relative_y = %d' %(relative_x, relative_y))
  if (relative_x) / float(width) >= 1 or (relative_y) / float(height) >= 1:
    raise ValueError('(relative_x) / float(width) >= 1 or (relative_y) / float(height) >= 1:'
                     ' (relative_x) / float(width) = %f (relative_y) / float(height) = %f'
                     %((relative_x) / float(width), (relative_y) / float(height)))
  return relative_x / float(width), relative_y / float(height)

def get_bounding_straight_rectangle(points):
  """Given a list of points, determine the straight rectangle bounding all of them. Returns xywh."""
  xs = [point[0] for point in points]
  ys = [point[1] for point in points]
  xmin, xmax = (min(xs), max(xs))
  ymin, ymax = (min(ys), max(ys))
  return xmin, ymin, xmax-xmin, ymax-ymin


def im2gray(image):
  '''Turn images into grayscale.'''
  if len(image.shape) == 2:
    return image
  image = image.astype(np.float32)
  # Use the Conversion Method in This Paper:
  # [http://www.eyemaginary.com/Rendering/TurnColorsGray.pdf]
  if image.shape[-1] == 1:
    image_gray = image
  elif image.shape[-1] == 3:
    image_gray = np.dot(image, [[0.2989],[0.5870],[0.1140]])
  elif image.shape[-1] == 4:
    # May be inaccurate since we lose the a channel.
    image_gray = np.dot(image[...,:3], [[0.2989],[0.5870],[0.1140]])
  else:
    raise NotImplementedError
  return image_gray

def is_sketch(image, max_pixel_val=255.0, max_diff_gray_tolerance=0.05, bw_tolerance=0.05, dark_percentage_upper_lim=0.2, bright_percentage_lower_lim=0.8):
  assert image.shape[-1] == 3 or image.shape[-1] == 1
  if image.shape[-1] == 3:
    gray = im2gray(image)
    # gray = np.expand_dims(image, axis=-1)
    max_diff_gray = np.max(gray - image)
    if max_diff_gray / max_pixel_val >= max_diff_gray_tolerance:
      return False
  else:
    gray = image
  # Now we are sure that the image is grayscale. The characteristic of a sketch is that there are two clusters of
  # colors, one around black and one around white.
  # median = np.median(gray)
  # if median / max_pixel_val < 1 - median_tolerance:
  #   return False

  dark_pixels = np.sum((gray / max_pixel_val) < bw_tolerance)
  bright_pixels = np.sum((gray / max_pixel_val) > (1 - bw_tolerance))
  total_num_pixels = gray.size

  dark_percentage = dark_pixels / float(total_num_pixels)
  bright_percentage = bright_pixels / float(total_num_pixels)
  del gray
  if dark_percentage > dark_percentage_upper_lim or bright_percentage < bright_percentage_lower_lim:
    return False
  return True


def pad_and_break_up_image(np_image, dest_height, dest_width):
  """Returns [num_image, dest_height, dest_width, channel], pad_height, pad_width."""
  image_height, image_width = get_image_height(np_image), get_image_width(np_image)
  def get_pad_val(input_dim, dest_dim):
    pad_dim = input_dim % dest_dim
    if pad_dim:
      pad_dim = dest_dim - pad_dim
    return pad_dim
  pad_h = get_pad_val(image_height, dest_height)
  pad_w = get_pad_val(image_width, dest_width)

  pad_val = 1.0 if np.issubdtype(np_image.dtype, np.floating) else 255
  padded = np.pad(np_image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant', constant_values=pad_val)
  num_h = get_image_height(padded) / dest_height
  num_w = get_image_width(padded) / dest_width
  ret = np.reshape(padded, (dest_height, num_h, dest_width, num_w, np_image.shape[-1]))
  ret = np.transpose(ret, (1, 3, 0, 2, 4))
  ret = np.reshape(ret, (num_h * num_w, dest_height, dest_width, np_image.shape[-1]))

  return ret, num_h, num_w, pad_h, pad_w

def regroup_broken_image(np_image, num_h, num_w, pad_h, pad_w):
  dest_height, dest_width= get_image_height(np_image), get_image_width(np_image)
  ret = np.reshape(np_image, (num_h, num_w, dest_height, dest_width, np_image.shape[-1]))
  ret = np.transpose(ret, (2, 0, 3, 1, 4))  # (dest_height, num_h, dest_width, num_w, np_image.shape[-1])
  ret = np.reshape(ret, (dest_height * num_h, dest_width * num_w, np_image.shape[-1]))
  return ret[:get_image_height(ret)-pad_h, :get_image_width(ret)-pad_w]
