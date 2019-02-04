import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from glob import glob
import PIL
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm_notebook
from matplotlib.image import imread
from sklearn.metrics import confusion_matrix


rd_bikes = list(Path(os.path.join(os.getcwd(), 'road_bikes')).glob('*jpg'))
mn_bikes = list(Path(os.path.join(os.getcwd(), 'mountain_bikes')).glob('*jpg'))
test_data = rd_bikes[80:] + mn_bikes[80:]
random.Random(42).shuffle(test_data)


def one_hot(label):
  temp = np.zeros((2))
  if label=='road':
    temp[0]=1
  elif label=='mountain':
    temp[1]=1
  return temp

x_test = np.zeros((test_data.__len__(), 224, 224, 3))
y_test = np.zeros((test_data.__len__(), 2))


for idx, path in tqdm(enumerate(test_data)):
  temp = PIL.Image.open(str(path))
  x_test[idx, :, :, :] = np.asarray(temp.resize((224, 224)))
  y_test[idx, :] = one_hot(path.stem.split('_')[0])

with tf.Session(graph=tf.Graph()) as sess:
	tf.saved_model.loader.load(sess, [tag_constants.TRAINING], os.getcwd())


