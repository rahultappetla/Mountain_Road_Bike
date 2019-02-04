from PIL import Image
import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from glob import glob
import PIL
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from matplotlib.image import imread
from sklearn.metrics import confusion_matrix


rd_bikes = list(Path(os.path.join(os.getcwd(), 'road_bikes')).glob('*jpg'))
mn_bikes = list(Path(os.path.join(os.getcwd(), 'mountain_bikes')).glob('*jpg'))
train_data = rd_bikes[:80] + mn_bikes[:80]
test_data = rd_bikes[80:] + mn_bikes[80:]
random.Random(42).shuffle(train_data)
random.Random(42).shuffle(test_data)


def one_hot(label):
  temp = np.zeros((2))
  if label=='road':
    temp[0]=1
  elif label=='mountain':
    temp[1]=1
  return temp

x_train = np.zeros((train_data.__len__(), 224, 224, 3))
y_train = np.zeros((train_data.__len__(), 2))


for idx, path in tqdm(enumerate(train_data)):
  temp = PIL.Image.open(str(path))
  x_train[idx, :, :, :] = np.asarray(temp.resize((224, 224)))
  y_train[idx, :] = one_hot(path.stem.split('_')[0])
  


n_classes = 2
batch_size = 8

x = tf.placeholder('float', [None, 224, 224, 3])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([7,7,3,32])),
               'W_conv2':tf.Variable(tf.random_normal([7,7,32,32])),
               'W_conv3':tf.Variable(tf.random_normal([7,7,32,64])),
               'W_conv4':tf.Variable(tf.random_normal([7,7,64,64])),
               'W_fc':tf.Variable(tf.random_normal([14*14*64,16])),
               'out':tf.Variable(tf.random_normal([16, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([32])),
              'b_conv3':tf.Variable(tf.random_normal([64])),
               'b_conv4':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([16])),
               'out':tf.Variable(tf.random_normal([n_classes]))}


    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    conv1 = tf.layers.batch_normalization(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    conv2 = tf.layers.batch_normalization(conv2)
    
    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = maxpool2d(conv3)
    conv3 = tf.layers.batch_normalization(conv3)

    conv4 = tf.nn.relu(conv2d(conv3, weights['W_conv4']) + biases['b_conv4'])
    conv4 = maxpool2d(conv4)
    conv4 = tf.layers.batch_normalization(conv4)
    
    

    fc = tf.reshape(conv4,[-1, 14*14*64])

    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']
    

    return output


def train_neural_network(x):
    
    logs_path = os.path.join(os.getcwd())
    with tf.name_scope('Model'):
    
      prediction = convolutional_neural_network(x)
    with tf.name_scope('Loss'):
      
      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    with tf.name_scope('SGD'):
      optimizer = tf.train.AdamOptimizer().minimize(cost)
     
    tf.summary.scalar("loss", cost)
    merged_summary_op = tf.summary.merge_all()
    
    hm_epochs = 1
    with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.summary.FileWriter('./graphs', graph=tf.get_default_graph())
        saver = tf.train.Saver()
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(8):
                epoch_x, epoch_y = x_train[i*20:i*20+20], y_train[i*20:i*20+20]
               
                _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c/8
                summary_writer.add_summary(summary, epoch*8+i)
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        con_mat = tf.confusion_matrix(tf.argmax(y, 1),
                                      tf.argmax(prediction, 1))
        print("Confusion Matrix for the Training Dataset: ")
        train_confusion = sess.run(con_mat, feed_dict={x:x_train, 
                                                      y:y_train})
        print(train_confusion)

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Train Accuracy:',accuracy.eval({x:x_train, y:y_train}))
        save_path = saver.save(sess, os.path.join(os.getcwd(), "model.ckpt"))
        print("Model saved in path: {0}".format(save_path))
train_neural_network(x)
