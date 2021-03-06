# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with mnist model.

The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import os
import sys
import threading
import time

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations  # pip install grpcio
import numpy as np
import tensorflow as tf
import scipy.misc

from tensorflow_serving.apis import predict_pb2   # pip install tensorflow-serving-api
from tensorflow_serving.apis import prediction_service_pb2

import util_io

tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('image_hw', 4, 'Height and width to resize the input image to.')
tf.app.flags.DEFINE_string('domain_translation_server', None, 'PredictionService host:port')
tf.app.flags.mark_flag_as_required('domain_translation_server')
FLAGS = tf.app.flags.FLAGS


# class _ResultCounter(object):
#   """Counter for the prediction results."""
#
#   def __init__(self, num_tests, concurrency):
#     self._num_tests = num_tests
#     self._concurrency = concurrency
#     self._error = 0
#     self._done = 0
#     self._active = 0
#     self._condition = threading.Condition()
#
#   def inc_error(self):
#     with self._condition:
#       self._error += 1
#
#   def inc_done(self):
#     with self._condition:
#       self._done += 1
#       self._condition.notify()
#
#   def dec_active(self):
#     with self._condition:
#       self._active -= 1
#       self._condition.notify()
#
#   def get_error_rate(self):
#     with self._condition:
#       while self._done != self._num_tests:
#         self._condition.wait()
#       return self._error / float(self._num_tests)
#
#   def throttle(self):
#     with self._condition:
#       while self._active == self._concurrency:
#         self._condition.wait()
#       self._active += 1
#
#
# def _create_rpc_callback_test(output_path, result_counter):
#   """Creates RPC callback function.
#
#   Args:
#     label: The correct label for the predicted example.
#     result_counter: Counter for the prediction result.
#   Returns:
#     The callback function.
#   """
#   def _callback(result_future):
#     """Callback function.
#
#     Calculates the statistics for the prediction result.
#
#     Args:
#       result_future: Result future of the RPC.
#     """
#     exception = result_future.exception()
#     if exception:
#       result_counter.inc_error()
#       print(exception)
#     else:
#       sys.stdout.write('.')
#       sys.stdout.flush()
#       response_images = np.reshape(np.array(
#           result_future.result().outputs['outputs'].float_val),
#         [dim.size for dim in result_future.result().outputs['outputs'].tensor_shape.dim])
#
#       assert response_images.shape[0] == 1  # TODO: temporary... modify this to support batch output.
#       for i in range(response_images.shape[0]):
#         util_io.imsave(output_path, response_images[i])
#
#     result_counter.inc_done()
#     result_counter.dec_active()
#   return _callback
#
#
# def do_inference_test(hostport, image_path, concurrency, num_tests, output_dir):
#   """Tests PredictionService with concurrent requests.
#
#   Args:
#     hostport: Host:port address of the PredictionService.
#     work_dir: The full path of working directory for test data set.
#     concurrency: Maximum number of concurrent requests.
#     num_tests: Number of test images to use.
#
#   Returns:
#     The classification error rate.
#
#   Raises:
#     IOError: An error occurred processing test data set.
#   """
#   image = np.expand_dims(util_io.imread(image_path, (FLAGS.image_hw, FLAGS.image_hw)) / 255.0, 0)
#   host, port = hostport.split(':')
#   channel = implementations.insecure_channel(host, int(port))
#   stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
#   result_counter = _ResultCounter(num_tests, concurrency)
#   for _ in range(num_tests):
#     request = predict_pb2.PredictRequest()
#     request.model_spec.name = 'test'
#     request.model_spec.signature_name = 'domain_transfer'
#     request.inputs['inputs'].CopyFrom(
#         tf.contrib.util.make_tensor_proto(image, shape=image.shape))
#     result_counter.throttle()
#     result_future = stub.Predict.future(request, 5.0)  # 5 seconds
#     result_future.add_done_callback(
#         _create_rpc_callback_test(output_dir, result_counter))
#     # result = stub.Predict(request, 5.0)  # 5 seconds
#     # print(result)
#     result_future.add_done_callback(
#         _create_rpc_callback_test(output_dir, result_counter))
#   return result_counter.get_error_rate()



class DomainTranslationClient():

  def __init__(self, hostport, image_hw, sample_images_path='', model_spec_name='test', model_spec_signature_name='domain_transfer', concurrency=1, ):
    """

    Args:
      hostport: Host:port address of the PredictionService.
      image_hw: Width and height of the input image.
      model_spec_name:
      model_spec_signature_name:
      concurrency: Maximum number of concurrent requests.
    """
    self.hostport = hostport
    self.image_hw = image_hw
    self.model_spec_name = model_spec_name
    self.model_spec_signature_name = model_spec_signature_name
    self.concurrency = concurrency
    # TODO: deprecate sample_images_path.
    self.sample_images_path = sample_images_path
    if self.sample_images_path:
      self.sample_images = np.load(self.sample_images_path)

    try:
      host, port = hostport.split(':')
      self.channel = implementations.insecure_channel(host, int(port))
    except ValueError as e:
      tf.logging.error('Cannot parse hostport %s' %hostport)
    self.request_template = predict_pb2.PredictRequest()
    self.request_template.model_spec.name = model_spec_name
    self.request_template.model_spec.signature_name = model_spec_signature_name


  def _concat_with_random_images(self, input_image, num_additional_images=15):
    """Concat the input image(1,hw,hw,3) with 15 other images from self.sample_images. Used for batch norm bug."""
    num_sample_images = self.sample_images.shape[0]
    additional_images = self.sample_images[np.random.choice(num_sample_images, num_additional_images)]
    return np.concatenate((input_image, additional_images, ), axis=0)


  @staticmethod
  def _request_set_input_image(request, input_image):
    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(input_image))

  @staticmethod
  def _create_rpc_callback(output_path):
    """Creates RPC callback function.

    Args:
      label: The correct label for the predicted example.
      result_counter: Counter for the prediction result.
    Returns:
      The callback function.
    """

    def _callback(result_future):
      """Callback function.

      Calculates the statistics for the prediction result.

      Args:
        result_future: Result future of the RPC.
      """
      exception = result_future.exception()
      if exception:
        print(exception)
      else:
        sys.stdout.write('.')
        sys.stdout.flush()
        # TODO: do post processing using another function.
        response_images = np.reshape(np.array(
          result_future.result().outputs['outputs'].float_val),
          [dim.size for dim in result_future.result().outputs['outputs'].tensor_shape.dim]) * 255.0

        util_io.imsave(output_path, response_images[0])
        # assert response_images.shape[0] == 1  # TODO: temporary... modify this to support batch output.
        # for i in range(response_images.shape[0]):
        #   util_io.imsave(output_path, response_images[i])

    return _callback

  def do_inference(self, output_dir, image_path=None, image_np=None):
    """Tests PredictionService with concurrent requests.

    Args:
      output_dir: Directory to output image.
      image_path: Path to image.

    Returns:
      `output_dir`.
    """
    if image_path is None and image_np is None:
      raise ValueError('Either `image_np` or `image_path` must be specified.')

    if image_path:
      image_resized = util_io.imread(image_path, (self.image_hw, self.image_hw))
    else:
      image_resized = scipy.misc.imresize(image_np, (self.image_hw, self.image_hw))
    # TODO: do preprocessing in a separate function. Check whether image has already been preprocessed.
    image = np.expand_dims(image_resized / np.float32(255.0), 0)
    # # TODO: temporary solution:
    # image_with_aux_batch = self._concat_with_random_images(image)
    image_with_aux_batch = image

    stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
    request = predict_pb2.PredictRequest()
    request.CopyFrom(self.request_template)
    self._request_set_input_image(request, image_with_aux_batch)
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(self._create_rpc_callback(output_dir))
    return output_dir

  def block_on_callback(self, output_dir):
    while not os.path.exists(output_dir):
      time.sleep(0.001)


def main(_):
  print("""Another way to test the inference model: 
        saved_model_cli run --dir 'path/to/export/model' \
        --tag_set serve  --signature_def serving_default --input_exprs 'inputs=np.ones((1,4,4,3))'""")
  if FLAGS.num_tests > 10000:
    print('num_tests should not be greater than 10k')
    return
  if not FLAGS.domain_translation_server:
    print('please specify domain_translation_server host:port')
    return
  util_io.touch_folder(FLAGS.output_dir)
  img_basename = os.path.basename(FLAGS.image_path)

  # client = DomainTranslationClient(FLAGS.domain_translation_server, FLAGS.image_hw, concurrency=FLAGS.concurrency, sample_images_path='/home/jerryli27/PycharmProjects/image2tag/data/inference_test_data/images_sample.npy')
  client = DomainTranslationClient(FLAGS.domain_translation_server, FLAGS.image_hw, concurrency=FLAGS.concurrency, sample_images_path='')
  client.do_inference(FLAGS.image_path, os.path.join(FLAGS.output_dir, img_basename))
  print('\nDone')


if __name__ == '__main__':
  tf.app.flags.DEFINE_string('image_path', '', 'Path to the image to be translated.')
  tf.app.flags.DEFINE_string('output_dir', '../data/face_translated/', 'Path to output directory for translated images.')
  tf.app.run()
