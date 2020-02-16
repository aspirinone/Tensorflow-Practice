import os
import sys
import tarfile

import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
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

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image = cv.imread("D:/tensorflow/models/research/object_detection/test_images/image1.jpg")
        #ret,image = capture.read()
        print(image.shape)
        image_np_expanded = np.expand_dims(image,axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes,scores,classes,num_detections) = sess.run([boxes,scores,classes,num_detections],
                                                         feed_dict={image_tensor:image_np_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            #min_score_thresh=0.8,
            use_normalized_coordinates=True,
            line_thickness=4
        )

        cv.imshow("SSD - object detection image",image)
        cv.waitKey(0)
        cv.destroyAllWindows()