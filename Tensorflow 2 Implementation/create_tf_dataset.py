import scipy.io as sio
import numpy as np
import os
import tensorflow as tf

def create_tf_dataset(
    d_index,d_type,fold,label_encoding='label'):
  """Create dataset from numpy array using tf.data.Dataset.from_tensor_slices
    
  d_index (int): index of dataset, 1 for ASL, 2 for ASL with digits, 3 for NUS
  
  d_type (string): a string to determine whether to use dataset with or 
      without augment data, "original" for dataset without augmented data, 
      else use dataset with augmented data
      
  fold (int): 1 to 5, specify which fold to use
  
  label_encoding (string): choose either 'label' or one-hot encoding to use, 
      default input for the label encoding is one-hot vector. If one-hot 
      encoding is used, the loss object and train and test accuracy must change
      to tf.keras.losses.CategoricalCrossentropy().

  Please make sure os is in the correct directory, which has folder containing 
  datasets, in this case CrossVal with datasets 1 2 3 in it
  """

  if fold == 1:
    file_name = '1st_fold'
  elif fold == 2:
    file_name = '2nd_fold'
  elif fold == 3:
    file_name = '3rd_fold'
  elif fold == 4:
    file_name = '4th_fold'
  elif fold == 5:
    file_name = '5th_fold'
  
  # load images and labels from CrossVal folder
  path = os.path.join('CrossVal', 'D'+str(d_index))
  print("Debugging, path: " ,path)
  if d_type == 'original':
    Train =sio.loadmat(os.path.join(path, 'D'+str(d_index)+'_'+file_name+'_train.mat'))
  else:
    Train =sio.loadmat(os.path.join(path, 'Augmented_D'+str(d_index)+'_'+file_name+'_train.mat'))
  Test = sio.loadmat(os.path.join(path, 'D'+str(d_index)+'_'+file_name+'_test.mat'))

  # assign images and labels to variables
  Train_Images = Train['trainImages']
  Train_Labels = Train['trainLabels2']
  Test_Images = Test['testImages']
  Test_Labels = Test['testLabels2']

  # Train_Images are in the shape of [height, width, batch_size]
  # the following command transpose it into [batch_size, height, width]
  x_train = Train_Images.transpose([2,0,1])
  x_test = Test_Images.transpose([2,0,1])

  # Labels are one-hot encoding and in the shape [num_classes, batch_size]
  # tf.keras.losses.CategoricalCrossentropy expects shape of 
  # [batch_size, num_classes], so tranpose is needed
  y_train = Train_Labels.transpose()
  y_test = Test_Labels.transpose()
  # Convert labels from one hot encoding to label encoding (represented by int)
  if label_encoding == 'label':
    y_train = tf.math.argmax(y_train, axis=1, output_type=tf.dtypes.int32)
    y_train = tf.cast(y_train, tf.dtypes.int8)
    y_test = tf.math.argmax(y_test, axis=1, output_type=tf.dtypes.int32)
    y_test = tf.cast(y_test, tf.dtypes.int8)

  # Add a dimension for grayscale images in training and testing set, from 
  # [batch_size, height, width] to [batch_size, height, width, 1]
  x_train = x_train[..., tf.newaxis].astype("float32")
  x_test = x_test[..., tf.newaxis].astype("float32")

  # The datasets are already shuffled, you can do without shuffling below.
  # For perfect shuffling, set the buffer size equal to the full size of 
  # the dataset.
  # see https://www.tensorflow.org/api_docs/python/tf/data/experimental/shuffle_and_repeat#:~:text=For%20perfect%20shuffling%2C%20set%20the,1%2C000%20elements%20in%20the%20buffer.
  if d_type == 'original':
    shuffle_buffer_size = x_train.shape[0]
  else: 
    if d_index == 1:
      # augmented dataset 1 has 500,000+ imgs for training, might be too big
      shuffle_buffer_size = x_train.shape[0] // 10
    else:
      # augmented dataset 2, 3 has 20,000+ and 16,000 imgs for training, 
      # might be okay to use the size of entire training set as buffer size 
      # for shuffling
      shuffle_buffer_size = x_train.shape[0]
  train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
      shuffle_buffer_size, seed=1)

  test_ds = tf.data.Dataset.from_tensor_slices(
      (x_test, y_test))

  return train_ds, test_ds