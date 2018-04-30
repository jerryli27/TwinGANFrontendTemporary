#!/usr/bin/env python
# -*- coding: utf-8 -*-
import util_misc

import os
import numpy as np
import tensorflow as tf

import util_io


class UtilMiscTest(tf.test.TestCase):
  def test_visualize_landmark_predictions(self):
    num_landmarks = 5
    image_size = 4
    landmark_predictions_np = np.zeros((2, 4, 4, num_landmarks, 3), dtype=np.float32)
    for i in range(num_landmarks):
      landmark_predictions_np[0, 0, 0, i, 2] = 100.0
      landmark_predictions_np[1, 1, 1, i, 2] = 10.0
    landmark_predictions = tf.constant(landmark_predictions_np)



    grid_size = landmark_predictions.shape[1]
    assert grid_size == landmark_predictions.shape[2]
    # Row = [[0, 1, 2, 3...], [0, 1, 2, 3...]...]. Column = [[0, 0, ..], [1, 1,...]...]
    row, column = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    dtype = landmark_predictions.dtype
    row = tf.cast(row, dtype) / tf.cast(grid_size, dtype)
    column = tf.cast(column, dtype) / tf.cast(grid_size, dtype)
    rc = tf.stack((row, column), axis=-1)
    rc_reshaped = tf.expand_dims(tf.expand_dims(rc, 0), 3)
    rc_reshaped = util_misc.tf_repeat(rc_reshaped, [landmark_predictions.shape[0], 1, 1, landmark_predictions.shape[3], 1])
    if True:
      landmark_confidence_reshaped = tf.transpose(landmark_predictions[..., 2], (0, 3, 1, 2))
      landmark_confidence_reshaped = tf.reshape(landmark_confidence_reshaped,
                                                tf.TensorShape([landmark_confidence_reshaped.shape[0],
                                                                landmark_confidence_reshaped.shape[1],
                                                                landmark_confidence_reshaped.shape[2] *
                                                                landmark_confidence_reshaped.shape[3], ]))
      c = util_misc.layers.softmax(landmark_confidence_reshaped)
    else:
      c = tf.sigmoid(landmark_predictions[..., 2])
    c = tf.transpose(c, (0, 2, 1))
    # c = tf.reshape(c, tf.TensorShape(
    #   [c.shape[0], grid_size, grid_size, c.shape[2]]))  # [batch size, height, width, num_landmarks]
    xy = rc_reshaped + landmark_predictions[..., 0:2] / int(
      grid_size)  # [batch size, height, width, num_landmarks, 2]
    xy_reshaped = tf.reshape(xy, tf.TensorShape([xy.shape[0], xy.shape[1] * xy.shape[2] * xy.shape[3] * xy.shape[4]]))
    xy_one_hot = util_misc.xy_to_one_hot(xy_reshaped, image_size)
    xy_one_hot_weighed = xy_one_hot * tf.expand_dims(tf.expand_dims(util_misc.layers.flatten(c), 1), 2)
    xy_one_hot_weighed = tf.reduce_sum(xy_one_hot_weighed, axis=-1, keepdims=True)






    landmark_visualization = xy_one_hot_weighed
    with self.test_session():
      actual_output = landmark_visualization.eval()
      print(actual_output)

  def test_is_sketch(self):
    image_dir = u'/mnt/f032b8a5-c186-4fae-b911-bcfdee99a2e9/pixiv_collected_sketches/PixivUtil2/test_samples_tiny'
    image_paths_and_expected = [
      (u'11739035_p1 - レミリア.jpg', False),
      (u'12925016_p2 - 東方系ラクガキ詰め合わせ.jpg', False),
      (u'14362806_p0 - 無題.jpg', False),  # Background not white enough.
      (u'14485122_p0 - 宮子.jpg', False),  # Background not white enough.
      (u'15444948_p4 - ぐ～てん☆もるげんっ！.jpg', False),
      (u'15469173_p0 - 東方魔理沙　塗ってみた.jpg', False),
      (u'17834866_p0 - 以蔵の闘い.jpg', False),
      (u'24774862_p2 - からくりばーすと、メイキング.jpg', False),
      (u'28474957_p0 - 白澤さん。.jpg', True),  # Wierd edge case unrecognizable to human eyes.
      (u'28592442_p0 - 本田菊.jpg', False),
      (u'29908672_p0 - ソードアート・おっぱい.jpg', False),
      (u'30454124_p0 - 水野亜美(セーラーマーキュリー).jpg', False),
      (u'5833646_p0 - Q命病棟でQP化(線画).png', True),
      (u'5943678_p0 - バニー達は塗ってほしそうにこちらを見ている・・.jpg', True),
      (u'5952556_p0 - 死神線画.png', True),
      (u'5981054_p0 - リン・レン線画.jpg', False),
      (u'5987994_p0 - お願いします。.png', True),
      (u'7210533_p0 - カウントダウン2日前.jpg', False),
      (u'7242346_p0 - ミリー【線画】.jpg', True),
      (u'7252006_p0 - ナタネ＆ロズレイド線画.jpg', True),
      (u'7304820_p0 - 少女の宮サマ。.jpg', True),
      (u'7431716_p0 - 今吉さん線画.jpg', True),
      (u'8113468_p0 - あんこくのじょおう【再投稿】.png', True),
      (u'8152845_p0 - ヘルメスさん.jpg', False),
      (u'8425348_p0 - 早苗さん（線画）.png', True),  # PNG bug, minor recall loss.
      (u'8441291_p0 - 【線画】ベルベル.jpg', False),
      (u'8502938_p0 - ベアト線画！.jpg', True),
    ]
    for image_name, expected in image_paths_and_expected:
      image = util_io.imread(os.path.join(image_dir, image_name))
      actual_output = util_misc.is_sketch(image)
      if not actual_output == expected:
        print('unexpected: ', image_name)
      # self.assertAllEqual(actual_output, expected)


  def test_pad_and_break_up_image(self):
    height, width, channel = 16, 16, 1
    image = np.array([[[h * w * (c + 1) for c in range(channel)] for w in range(width)] for h in range(height)], dtype=np.uint8)
    dest_height, dest_width = 3, 4
    broken_image, num_h, num_w, pad_h, pad_w = util_misc.pad_and_break_up_image(image, dest_height, dest_width)
    # Not the best way to test it, but works for now..
    regrouped_image = util_misc.regroup_broken_image(broken_image, num_h, num_w, pad_h, pad_w)
    self.assertAllEqual(regrouped_image, image)

if __name__ == '__main__':
  tf.test.main()
