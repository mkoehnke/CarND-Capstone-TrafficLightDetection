# Based on https://github.com/tensorflow/models/blob/master/object_detection/create_pascal_tf_record.py

"""Convert traffic light dataset to TFRecord for object_detection.
Example usage:
    ./create_traffic_light_tf_record --data_dir=/home/user/tl --output_path=/home/user/tl.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import glob
from io import BytesIO

from lxml import etree
import PIL.Image
import tensorflow as tf
import random

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw traffic light dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', 'data/traffic_dataset.record', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'model/label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']

def convert_to_jpeg(im):
    with BytesIO() as f:
        im.save(f, format='JPEG', quality=95)
        return f.getvalue()

def dict_to_tf_example(data,
                       xml_path,
                       label_map_dict,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.
  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.
  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    xml_path: Path to xml file
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
  Returns:
    example: The converted tf.Example.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  full_path = os.path.join(os.path.dirname(xml_path), data['filename'])
  print("FULL %s" % full_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  image = PIL.Image.open(io.BytesIO(encoded_jpg))
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  
  width = int(data['size']['width'])
  height = int(data['size']['height'])
  target_width = 300
  target_height = 225 
  print("WIDTH: %d HEIGHT: %d, target_width: %d, target_height: %d" % (width, height, target_width, target_height))
  image = image.resize((target_width, target_height))
  encoded_jpg = convert_to_jpeg(image)

  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))
    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(target_height),
      'image/width': dataset_util.int64_feature(target_width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      # 'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      # 'image/object/truncated': dataset_util.int64_list_feature(truncated),
      # 'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))

  data_dir = FLAGS.data_dir

  train_writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  validation_writer = tf.python_io.TFRecordWriter(FLAGS.output_path + ".validation")

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from dataset.')
  annotations_dir = data_dir
  files = glob.glob(os.path.join(data_dir, "*.xml"))
  random.shuffle(files)
  train_files = files[:int(len(files)*0.8)]
  validation_files = files[int(len(files)*0.8):]
  for writer, files in [(train_writer, train_files), (validation_writer, validation_files)]:
    for idx, xml_file in enumerate(files):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(files))
      path = xml_file
      with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

      tf_example = dict_to_tf_example(data, path, label_map_dict, FLAGS.ignore_difficult_instances)
      writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
  tf.app.run()
