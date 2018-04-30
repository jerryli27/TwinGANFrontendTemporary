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

from __future__ import print_function

import shutil
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

FLAGS = tf.app.flags.FLAGS

class SketchRefinementClient():

  def __init__(self, hostport, image_hw, supervised=False, model_spec_name='test', model_spec_signature_name=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY, concurrency=1, timeout=5.0):
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
    self.supervised = supervised
    self.model_spec_name = model_spec_name
    self.model_spec_signature_name = model_spec_signature_name
    self.concurrency = concurrency
    self.timeout=timeout

    try:
      host, port = hostport.split(':')
      self.channel = implementations.insecure_channel(host, int(port))
    except ValueError as e:
      tf.logging.error('Cannot parse hostport %s' %hostport)
    self.request_template = predict_pb2.PredictRequest()
    self.request_template.model_spec.name = model_spec_name
    self.request_template.model_spec.signature_name = model_spec_signature_name

  @staticmethod
  def _request_set_input_image(request, input_image):
    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(input_image))

  @staticmethod
  def _create_rpc_callback(output_path, sketch_image, supervised=False, **kwargs):
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

        if supervised:
          start = kwargs['start']
          end = kwargs['end']
          subregion_shape = kwargs['subregion_shape']
          # Use of flag here may cause some bug...
          if subregion_shape[0] != FLAGS.image_hw or subregion_shape[1] != FLAGS.image_hw:
            subregion = scipy.misc.imresize(response_images[0][...,0], (subregion_shape[0], subregion_shape[1]))
            subregion = np.expand_dims(subregion, -1)
          else:
            subregion = response_images[0]
          output = sketch_image.copy()
          output[start[0]:end[0], start[1]:end[1]] = subregion
        else:
          output = response_images[0]

        # It seems during imsave and before save finishes, it is possible to read the half-written file.
        # To prevent those bugs, I write it to a temporary file and move it after I am done.
        temporary_file = os.path.splitext(output_path)[0] + '.tmp' + os.path.splitext(output_path)[1]
        util_io.imsave(temporary_file, output)
        shutil.move(temporary_file, output_path)
        # assert response_images.shape[0] == 1  # TODO: temporary... modify this to support batch output.
        # for i in range(response_images.shape[0]):
        #   util_io.imsave(output_path, response_images[i])

    return _callback

  def do_inference(self, output_dir, center_point_xy, sketch_image_np=None, image_path=None, image_np=None):
    """Tests PredictionService with concurrent requests.

    Args:
      output_dir: Directory to output image.
      image_path: Path to image.

    Returns:
      `output_dir`.
    """
    sketch_image = sketch_image_np
    if len(sketch_image.shape) == 2:
      sketch_image = np.expand_dims(sketch_image, axis=-1)
    if self.supervised:
      if image_path is None and image_np is None:
        raise ValueError('Either `image_np` or `image_path` must be specified.')

      if image_path:
        image = util_io.imread(image_path, bw=True)
      else:
        image = image_np
      if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
      assert image.shape == sketch_image.shape
      combined = np.concatenate((sketch_image, image), axis=-1).astype(np.float32)
      # Select the subregion of interest.
      # Note: In numpy and tensorflow we're in (h,w,c) format.
      start = (max(0, center_point_xy[1] - self.image_hw / 2), max(0, center_point_xy[0] - self.image_hw / 2))
      end = (center_point_xy[1] + self.image_hw / 2, center_point_xy[0] + self.image_hw / 2)
      subregion = combined[start[0]:end[0], start[1]:end[1]]
      subregion_shape = subregion.shape
      if subregion_shape[0] != self.image_hw or subregion_shape[1] != self.image_hw:
        # Stupid imresize only accepts hxw images or hxwx3 images.
        subregion_resized = np.concatenate((np.expand_dims(scipy.misc.imresize(subregion[..., 0], (self.image_hw, self.image_hw)), axis=-1),
                                            np.expand_dims( scipy.misc.imresize(subregion[..., 1], (self.image_hw, self.image_hw)), axis=-1)),
                                           axis=-1)
      else:
        subregion_resized = subregion

      # TODO: do preprocessing in a separate function. Check whether image has already been preprocessed.
      subregion_resized = np.expand_dims(subregion_resized / np.float32(255, ), 0)
      input_image = subregion_resized
      callback_kwargs = {'start':start, 'end':end, 'subregion_shape':subregion_shape}
    else:
      input_image = sketch_image
      callback_kwargs = dict()

    stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
    request = predict_pb2.PredictRequest()
    request.CopyFrom(self.request_template)
    self._request_set_input_image(request, input_image)
    result_future = stub.Predict.future(request, self.timeout)  # 5 seconds
    result_future.add_done_callback(self._create_rpc_callback(output_dir, sketch_image, supervised=self.supervised, **callback_kwargs))
    return output_dir

  def block_on_callback(self, output_dir):
    start_time = time.time()
    while not os.path.exists(output_dir) and time.time() - start_time <= self.timeout:
      time.sleep(0.01)
    return os.path.exists(output_dir)


def main(_):
  print("""Another way to test the inference model: 
        saved_model_cli run --dir 'path/to/export/model' \
        --tag_set serve  --signature_def serving_default --input_exprs 'inputs=np.ones((1,4,4,3))'""")
  util_io.touch_folder(FLAGS.output_dir)
  img_basename = os.path.basename(FLAGS.image_path)

  # client = DomainTranslationClient(FLAGS.domain_translation_server, FLAGS.image_hw, concurrency=FLAGS.concurrency, sample_images_path='/home/jerryli27/PycharmProjects/image2tag/data/inference_test_data/images_sample.npy')
  client = SketchRefinementClient(FLAGS.sketch_refinement_server, FLAGS.image_hw, concurrency=FLAGS.concurrency)
  output_dir = os.path.join(FLAGS.output_dir, img_basename)
  client.do_inference(output_dir, center_point_xy=[14, 14], image_path=FLAGS.image_path, sketch_image_path=FLAGS.image_path)
  client.block_on_callback(output_dir)
  print('\nDone')


if __name__ == '__main__':
  tf.app.flags.DEFINE_string('sketch_refinement_server', None, 'PredictionService host:port')
  tf.app.flags.mark_flag_as_required('sketch_refinement_server')
  tf.app.flags.DEFINE_integer('concurrency', 1,
                              'maximum number of concurrent inference requests')
  tf.app.flags.DEFINE_integer('image_hw', 4, 'Height and width to resize the input image to.')
  tf.app.flags.DEFINE_string('image_path', '', 'Path to the image to be translated.')
  tf.app.flags.DEFINE_string('output_dir', '../data/face_translated/', 'Path to output directory for translated images.')
  tf.app.run()
