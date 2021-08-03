import tensorflow as tf
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, BatchNormalization,
GlobalAveragePooling2D, MaxPool2D, Dropout, ReLU, AveragePooling2D)
from tensorflow.keras import Model
from create_tf_dataset import create_tf_dataset
from EDenseNet_cls import EDenseNet as EDenseNet
print(tf.__version__)
seed_value= 1
tf.random.set_seed(seed_value)

    
dataset_index = 3
data_type = 'original' #'original' or 'augmented'
fold = 1
batch_size = 16
test_batch_size = 1024 # Use smaller value if GPU memory is small
epochs_to_train = 200
train_ds, test_ds = create_tf_dataset(dataset_index, data_type, fold)
train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)
# You can use the following code to see if you can get faster 
# training / testing time

# train_ds = train_ds.cache()
# train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
# test_ds = test_ds.cache()
# test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

class Model_Parameters:
  num_layers = 4
  growth_rate = 24
  filter_size = 3
  fm_1st_layer = 32
  bottleneck = 4
  dropout_prob = 0.2 # not 0.8, 0.8 drops 80% of the elements
  learn_rate = 0.001
  if dataset_index == 1:
    num_classes = 24
  elif dataset_index == 2:
    num_classes = 36
  else: 
    num_classes = 10
    
param = Model_Parameters()
model = EDenseNet(param, name="EDenseNet")
model = model.model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=param.learn_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print(model.summary())
# you can use model.get_weights() to check if setting random seed is working
# or not, you should see the same weights after you restart the runtime
print(model.get_weights())

history = model.fit(train_ds, epochs=epochs_to_train, 
                    validation_data=test_ds, verbose=1)
# validation=test_ds used in model.fit() is simply a shortcut for
# model.evaluate(test_ds), the model IS NOT BEING TRAINED on validation_data when you use it in model.fit(), 
# see official docs here, https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
# you can verify it by puting model.fit and model.evaluate in a for loop, and in each for loop,
# set epochs=1 in model.fit(), and to see the model's performance on test set, 
# simply by adding 
# test_loss, test_acc = model.evaluate(test_ds, verbose=2) afterwards.
# print(f'test loss: {test_loss}, test accuracy: {test_acc}')
# to keep track of test_loss and test_acc at the end of each epoch,
# declare a list before the for loop, 
# test_loss_list = []
# test_acc_list = []
# then append the test_loss and test_acc returns by model.evaluate to the list inside the for loop
# test_loss_list.append(test_loss)
# test_acc_list.append(test_acc)
# Then, when you plot the graph, comment out line 81, and uncomment line 82
# ALTERNATIVELY, you can simply just use validation_data=test_ds in model.fit()
                    
# plot graph for accuracy over epochs
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.plot(test_acc_list , label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
