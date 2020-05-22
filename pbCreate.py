import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np


def freeze_graph(input_checkpoint, output_graph):
    output_node_names = "input_x,output"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

# 输入ckpt模型路径
input_checkpoint='D:/TensorflowTest/taitanic/model/model.ckpt-9001'
# 输出pb模型的路径
out_pb_path="D:/TensorflowTest/taitanic/model/frozen_model.pb"
# 调用freeze_graph将ckpt转为pb
freeze_graph(input_checkpoint,out_pb_path)

