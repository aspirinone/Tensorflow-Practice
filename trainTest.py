import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

data = pd.read_csv('./data/train.csv')
#剔除影响较小的特征
data = data[[ 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch',  'Fare', 'Cabin', 'Embarked']]
#data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Cabin'] = pd.factorize(data.Cabin)[0]
data.fillna(0,inplace = True)
data['Sex'] = [1 if x == 'male' else 0 for x in data.Sex]
#使用独立编码，避免出现数值倍数关系
data['p1'] = np.array(data['Pclass'] == 1).astype(np.int32)
data['p2'] = np.array(data['Pclass'] == 2).astype(np.int32)
data['p3'] = np.array(data['Pclass'] == 3).astype(np.int32)
#然后删除pclass这行
del data['Pclass']
#print(data.Embarked.unique())
#处理港口号
data['e1'] = np.array(data['Embarked'] == 'S').astype(np.int32)
data['e2'] = np.array(data['Embarked'] == 'C').astype(np.int32)
data['e3'] = np.array(data['Embarked'] == 'Q').astype(np.int32)
del data['Embarked']
#print(data)

data_train = data[['Sex', 'Age', 'SibSp','Parch',  'Fare', 'Cabin', 'p1','p2','p3','e1','e2','e3']]
#这部分矩阵转换再考虑下
data_target = data['Survived'].values.reshape(len(data),1)

#以上，就是数据预处理的过程

#接下来是搭建神经网络预测部分
#这是待喂入的数据
x = tf.placeholder(shape=[None,12],dtype = tf.float32,name='input_x')
y = tf.placeholder(shape=[None,1],dtype = tf.float32,name='input_y')

weight = tf.Variable(tf.random.normal([12,1]))
bias = tf.Variable(tf.random.normal([1]))
output = tf.add(tf.matmul(x,weight),bias,name = 'output')

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=output))
step = tf.train.GradientDescentOptimizer(0.0003).minimize(loss)

pred = tf.cast(tf.sigmoid(output)>0.5,tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred,y),tf.float32))

data_test = pd.read_csv('./data/test.csv')
data_test = data_test[['Pclass', 'Sex', 'Age', 'SibSp','Parch',  'Fare', 'Cabin', 'Embarked']]
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].mean())
data_test['Cabin'] = pd.factorize(data_test.Cabin)[0]
data_test.fillna(0,inplace = True)
data_test['Sex'] = [1 if x == 'male' else 0 for x in data_test.Sex]
data_test['p1'] = np.array(data_test['Pclass'] == 1).astype(np.int32)
data_test['p2'] = np.array(data_test['Pclass'] == 2).astype(np.int32)
data_test['p3'] = np.array(data_test['Pclass'] == 3).astype(np.int32)
del data_test['Pclass']
data_test['e1'] = np.array(data_test['Embarked'] == 'S').astype(np.int32)
data_test['e2'] = np.array(data_test['Embarked'] == 'C').astype(np.int32)
data_test['e3'] = np.array(data_test['Embarked'] == 'Q').astype(np.int32)
del data_test['Embarked']

test_label = pd.read_csv('./data/gender.csv')
test_label = np.reshape(test_label.Survived.values.astype(np.float32),(418,1))

saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    loss_train = []
    train_acc = []
    test_acc = []
    sess.run(init)
    for i in range(10000):
        #这部分是在打乱顺序，防止过拟合
        #index = np.random.permutation(len(data_target))
        #data_train = data_train[index]
        #data_target = data_target[index]
        for n in range(len(data_target)//100+1):
            batch_xs = data_train[n*100:n*100+100]
            batch_ys = data_target[n * 100:n * 100 + 100]
            sess.run(step,feed_dict={x:batch_xs,y:batch_ys})
        if i%1000 == 0:
            loss_temp = sess.run(loss,feed_dict={x:batch_xs,y:batch_ys})
            loss_train.append(loss_temp)
            train_acc_temp = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys})
            train_acc.append(train_acc_temp)
            test_acc_temp = sess.run(accuracy,feed_dict={x:data_test,y:test_label})
            test_acc.append(test_acc_temp)
            print(loss_temp,train_acc_temp,test_acc_temp)
            saver.save(sess,'./model/model.ckpt',global_step=i+1)

plt.plot(loss_train,'k-')
plt.title('train loss')
plt.show()

plt.plot(train_acc,'b-',label = 'train_acc')
plt.plot(test_acc,'r--',label = 'test_acc')
plt.title('train and test accuracy')
plt.legend()
plt.show()




