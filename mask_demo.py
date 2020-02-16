import os
import sys
import tarfile

import cv2 as cv
import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
MODEL_FILE = 'D:/tensorflow/'+ MODEL_NAME + '.tar'

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('D:/tensorflow/models/research/object_detection/data','mscoco_label_map.pbtxt')

NUM_CLASSES = 90
#capture = cv.VideoCapture(0)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file,os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH,'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def,name = '')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categorys = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
category_index = label_map_util.create_category_index(categorys)

def load_image_into_numpy(image):
    (im_w,im_h) = image.size
    return np.array(image.getdata()).reshape(im_h,im_w,3).astype(np.uint8)

def run_inference_for_single_image(image,graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict ={}
            for key in [
                'num_detections','detection_boxes','detection_scores',
                'detection_classes','detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'],[0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0],tf.int32)
                detection_boxes = tf.slice(detection_boxes,[0,0],[real_num_detection,-1])
                detection_masks = tf.slice(detection_masks, [0, 0,0], [real_num_detection, -1,-1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks,detection_boxes,image.shape[0],image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed,0.5),tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed,0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor:np.expand_dims(image,0)})

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

image = cv.imread("D:/tensorflow/models/research/object_detection/test_images/image1.jpg")
print(image.shape)

output_dict = run_inference_for_single_image(image,detection_graph)

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks'),
    min_score_thresh=0.5,
    use_normalized_coordinates=True,
    line_thickness=4
)

cv.imshow("mask rcnn demo",image)
cv.waitKey(0)
cv.destroyAllWindows()
