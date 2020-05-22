# coding: utf-8
import tensorflow as tf
import pandas as pd
import os
import numpy as np
from tensorflow.python.platform import gfile


# 先检测看pb文件是否存在
savePbFile = './model/frozen_model.pb'
if os.path.exists(savePbFile) is False:
    print('Not found pb file!')
    exit()


data_test = pd.read_csv('./data/test.csv')
data_test = data_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].mean())
data_test['Cabin'] = pd.factorize(data_test.Cabin)[0]
data_test.fillna(0, inplace=True)
data_test['Sex'] = [1 if x == 'male' else 0 for x in data_test.Sex]
data_test['p1'] = np.array(data_test['Pclass'] == 1).astype(np.int32)
data_test['p2'] = np.array(data_test['Pclass'] == 2).astype(np.int32)
data_test['p3'] = np.array(data_test['Pclass'] == 3).astype(np.int32)
del data_test['Pclass']
data_test['e1'] = np.array(data_test['Embarked'] == 'S').astype(np.int32)
data_test['e2'] = np.array(data_test['Embarked'] == 'C').astype(np.int32)
data_test['e3'] = np.array(data_test['Embarked'] == 'Q').astype(np.int32)
del data_test['Embarked']

#print(data_test)


#data_test = data_test.astype()
gender = pd.read_csv('./data/gender.csv')
gender = np.reshape(gender.Survived.values.astype(np.float32),(418,1))


with tf.Session() as sess:
    # 打开pb模型文件
    with gfile.FastGFile(savePbFile, 'rb') as fd:
        # 导入图
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fd.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        # 根据名字获取对应的tensorflow
        input = sess.graph.get_tensor_by_name('input_x:0')
        output = sess.graph.get_tensor_by_name('output:0')
        output_e = tf.cast(tf.sigmoid(output)>0.5,tf.float32)
        acc = tf.reduce_mean(tf.cast(tf.equal(gender, output_e),dtype = tf.float32))
        result = sess.run(output_e, feed_dict={input:data_test})
        accuracy = sess.run(acc,feed_dict={input:data_test})
        print(result)
        print(accuracy)
