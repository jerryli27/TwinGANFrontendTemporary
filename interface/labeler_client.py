import os

import util_io
import util_misc
import interface_utils


class LabelerClient(object):
  def __init__(self,):
    self.image_paths = []
    self.sketch_paths = []
    self.index = 0
    self.done_image_paths = set()
    self.done_image_txt_path = ''
    self.sketch_folder = ''

  def set_image_paths(self, image_path, finished_image_txt_path, sketch_folder, exclude_file_start={'e', 'q'}):
    if image_path:
      self.image_paths = util_io.get_all_image_paths(image_path)
      # Danbooru specific method to filter out nsfw images.
      self.image_paths = [p for p in self.image_paths if os.path.basename(p)[0] not in exclude_file_start]
      self.sketch_paths = [None for _ in range(len(self.image_paths))]
      self.index = 0
    if finished_image_txt_path:
      self.done_image_txt_path = finished_image_txt_path
      dir = os.path.dirname(finished_image_txt_path)
      self.colored_sketch_pair_txt_path = os.path.join(dir, 'colored_sketch_pair.txt')
      util_io.touch_folder(dir)
      try:
        self.done_image_paths = set(util_io.get_all_image_paths(finished_image_txt_path))
      except AssertionError:
        pass
    self.sketch_folder = sketch_folder
    sketches = set([util_misc.get_no_ext_base(p) for p in util_io.get_all_image_paths(sketch_folder)])
    self.image_paths = [p for p in self.image_paths if util_misc.get_no_ext_base(p) in sketches]
    pass


  def get_image_and_id(self):
    """Returns an image encoded in base64."""
    while self.index < len(self.image_paths) and self.image_paths[self.index] in self.done_image_paths:
      self.index += 1
    if self.index == len(self.image_paths):
      return None, None, None

    image = interface_utils.get_image_encoding(self.image_paths[self.index])
    image_id = os.path.basename(self.image_paths[self.index])

    sketch_image_path = self.get_current_sketch_path()
    sketch = interface_utils.get_image_encoding(sketch_image_path)
    self.sketch_paths[self.index] = sketch_image_path
    return image, sketch, image_id

  def mark_current_as_done(self, is_skip):
    with open(self.done_image_txt_path, 'a') as f:
      f.write(self.image_paths[self.index] + '\n')
    if not is_skip:
      with open(self.colored_sketch_pair_txt_path, 'a') as f:
        f.write(self.image_paths[self.index]+'\t' + self.sketch_paths[self.index] + '\n')
    self.done_image_paths.add(self.image_paths[self.index])
    self.index += 1

  def get_current_sketch_path(self):
    return self.get_sketch_path_for_image_name(self.image_paths[self.index])

  def get_sketch_path_for_image_name(self, image_basename):
    return os.path.join(self.sketch_folder, util_misc.get_no_ext_base(image_basename) + '.jpg')

  def set_current_sketch_path(self, new_path):
    self.sketch_paths[self.index] = new_path