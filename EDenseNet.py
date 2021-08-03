def densenet(D,DT,F,model):
  import scipy.io as sio
  import time
  import os
  import math
  import numpy as np
  import matplotlib.pyplot as plt


  Dataset = D
  if DT == 'org':
    data_type = 'original'
  else:
    data_type = 'augmented'

  fs = model.fs
  fm1 = model.fm1
  batch_size = model.batch_size[0] 
  learn_rate = model.learn_rate
  num_layers = model.num_layers
  k_fm = model.k_fm
  bottleneck = model.bottleneck
  dropout_prob = model.dropout_prob
  num_of_test = model.num_of_test

  ###############
  # load training / testing set from CrossVal folder,
  # names for training set, 'D1_1st_fold_train.mat', 'Augmented_D1_1st_fold_train.mat'
  # name for testing set, 'D1_1st_fold_test.mat'
  ###############
  if F == 1:
    file_name = '1st_fold'
  elif F == 2:
    file_name = '2nd_fold'
  elif F == 3:
    file_name = '3rd_fold'
  elif F == 4:
    file_name = '4th_fold'
  elif F == 5:
    file_name = '5th_fold'
  path = os.path.join('CrossVal', 'D'+Dataset)
  print("path " ,path)
  if data_type == 'original':
    Train =sio.loadmat(os.path.join(path, 'D'+Dataset+'_'+file_name+'_train.mat'))
  else:
    Train =sio.loadmat(os.path.join(path, 'Augmented_D'+Dataset+'_'+file_name+'_train.mat'))
  Test = sio.loadmat(os.path.join(path, 'D'+Dataset+'_'+file_name+'_test.mat'))

  if Dataset == '1':
    number_of_classes = 24
    num_of_ep = 50
    num_of_test = 20
    if data_type == 'augmented':
      train_imgs = 526190
    else:
      train_imgs = 52619
    iteration = math.ceil((num_of_ep * train_imgs) / batch_size)
  elif Dataset == '2':
    number_of_classes = 36
    num_of_ep = 200
    if data_type == 'augmented':
      train_imgs = 20120
    else:
      train_imgs = 2012
    iteration = math.ceil((num_of_ep * train_imgs) / batch_size)
  else:
    number_of_classes = 10
    num_of_ep = 200
    if data_type == 'augmented':
      train_imgs = 16000
    else:
      train_imgs = 1600
    iteration = math.ceil((num_of_ep * train_imgs) / batch_size)

  iteration_to_display = int(iteration / num_of_test) 
  list_to_display = []
  for i in range(num_of_test):
      if i !=num_of_test:
          list_to_display.append(int(iteration_to_display*(i+1)))
  del i


  total_fm_Block_1 = fm1+(num_layers*k_fm)
  total_fm_Block_2 = total_fm_Block_1+(num_layers*k_fm)
  total_fm_Block_3 = total_fm_Block_2+(num_layers*k_fm)
  fc_nodes = [total_fm_Block_3 ]


  Train_Images = Train['trainImages']
  Train_Labels = Train['trainLabels2']
  total_trainImages = len(Train_Images[0,2])
  print(total_trainImages)
  Train_Images = Train_Images.reshape(784,total_trainImages).transpose().astype('float32')
  Train_Labels = Train_Labels.transpose().astype('float64')


  Test_Images = Test['testImages']
  Test_Labels = Test['testLabels2']
  total_testImages = len(Test_Images[0,2])
  Test_Images = Test_Images.reshape(784,total_testImages).transpose().astype('float32')
  Test_Labels = Test_Labels.transpose().astype('float64')
  Target_labels = np.argmax(Test_Labels,axis=1)

  del Test
  del Train

  import tensorflow as tf
  tf.reset_default_graph()
  g = tf.Graph()
  with g.as_default():
    tf.set_random_seed(1)

    def weight_variable(shape,n):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial,name=n)

    def bias_variable(shape,n):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial,name=n)

    def avg_pool(input, s):
      return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'SAME')

    def max_pool(input, s):
      return tf.nn.max_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'SAME')

    def conv2d_1(input, in_features, out_features, kernel_size, name="W", with_bias=False):
      W = weight_variable([ kernel_size, kernel_size, in_features, out_features], name)
      conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
      if with_bias:
        return conv + bias_variable([ out_features ])
      return conv

    def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob, idx, scope='conv_block'):
      with tf.variable_scope(scope):
        current = tf.layers.batch_normalization(current, scale=True, training=is_training)
        current = tf.nn.relu(current)
        current = conv2d_1(current, in_features, out_features, kernel_size, name="W"+str(idx))
        current = tf.nn.dropout(current, keep_prob)
        return current

    def block(input, layers, in_features, growth, is_training, keep_prob, name="Block_"):
      with tf.name_scope(name):
        with tf.variable_scope(name):
          current = input
          features = in_features
          for idx in range(layers):
            tmp = batch_activ_conv(current, features, growth, fs, is_training, keep_prob, idx+1, scope='conv_block_'+str(idx+1))
            current = tf.concat((current, tmp), axis=3)
            features += growth
          return current, features


    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, number_of_classes])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)


    current = conv2d_1(x_image, 1, fm1, fs, name="W1", with_bias=False)

    current, features = block(current, num_layers, fm1, k_fm, training, keep_prob, name="Block_1")
    b1_conv_printop = tf.Print(current, [current])
    with tf.name_scope("transition_lyr"):
      #current = batch_activ_conv(current, features, features, 1, training, keep_prob, 1, scope='Transition_layer_1')
      current = batch_activ_conv(current, features, bottleneck*k_fm, 1, training, keep_prob, 1, scope='Transition_layer_1')
      t1_b_conv_printop = tf.Print(current, [current])
      current = batch_activ_conv(current, bottleneck*k_fm, features, fs, training, keep_prob, 1, scope='Transition_layer_1_1')
      t1_conv_printop = tf.Print(current, [current])
      current = max_pool(current, 2)
      #current = avg_pool(current, 2)
    current, features = block(current, num_layers, features, k_fm, training, keep_prob, name="Block_2")
    b2_conv_printop = tf.Print(current, [current])
    with tf.name_scope("transition_lyr_2"):
      #current = batch_activ_conv(current, features, features, 1, training, keep_prob, 1, scope='Transition_layer_2')
      current = batch_activ_conv(current, features, bottleneck*k_fm, 1, training, keep_prob, 1, scope='Transition_layer_2')
      t2_b_conv_printop = tf.Print(current, [current])
      current = batch_activ_conv(current, bottleneck*k_fm, features, fs, training, keep_prob, 1, scope='Transition_layer_2_1')
      t2_conv_printop = tf.Print(current, [current])
      current = max_pool(current, 2)
      #current = avg_pool(current, 2)
    current, features = block(current, num_layers, features, k_fm, training, keep_prob, name="Block_3")
    b3_conv_printop = tf.Print(current, [current])
    with tf.name_scope("transition_lyr_3"):
      #current = batch_activ_conv(current, features, features, 1, training, keep_prob, 1, scope='Transition_layer_3')
      current = batch_activ_conv(current, features, bottleneck*k_fm, 1, training, keep_prob, 1, scope='Transition_layer_3')
      t3_b_conv_printop = tf.Print(current, [current])
      current = batch_activ_conv(current, bottleneck*k_fm, features, fs, training, keep_prob, 1, scope='Transition_layer_3_1')
      t3_conv_printop = tf.Print(current, [current])
      current = avg_pool(current, 7)
      current = tf.reshape(current, [tf.shape(current)[0], -1])

    with tf.name_scope("Dense_Last_lyr"):
      W_fc3 = weight_variable([fc_nodes[0], number_of_classes],"w_fc3")
      b_fc3 = bias_variable([number_of_classes],"b_fc3")
      y_conv = tf.matmul(current, W_fc3) + b_fc3
      prediction_prob = tf.nn.softmax(y_conv)
      prediction_prob_printop = tf.Print(prediction_prob, [prediction_prob])

    with tf.name_scope("Xent"):
       cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    with tf.name_scope("train"):
      extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(extra_update_ops):
         train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      wrong_prediction = tf.not_equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      wrong_prediction_printop = tf.Print(wrong_prediction, [wrong_prediction])
      predicted_labels = tf.argmax(y_conv, 1)
      predicted_labels_printop = tf.Print(predicted_labels, [predicted_labels])

    index = 0
    index_end = index + batch_size
    remaining = 0
    start_time = time.time()
    costs = []
    accuracy_list = []
    list_of_predicted_list = []

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer(),tf.set_random_seed(0))
      for i in range(iteration):
        if index_end > total_trainImages:
                  remaining = total_trainImages - (index_end-batch_size)  
                  images = Train_Images[(index_end-batch_size):total_trainImages, :]
                  labels = Train_Labels[(index_end-batch_size):total_trainImages, :]
                  index = 0
                  index_end = index + batch_size - remaining
                  images = np.vstack((images, Train_Images[index:index_end, :]))
                  labels = np.vstack((labels, Train_Labels[index:index_end, :]))
                  batch = (images, labels)
                  index = index_end
                  index_end = index + batch_size
        else:
                  batch = (Train_Images[index:index_end, :], Train_Labels[index:index_end, :])
                  index = index + batch_size 
                  index_end = index_end + batch_size

        if i in list_to_display:
          elapsed_time = time.time() - start_time
          print('Elapsed Time Before for loop: %f secs' % elapsed_time)
          Accuracy = 0
          itrt_index = i
          print('debug: %d & %d' % (iteration,i))

          if Dataset == '1':
            if file_name == '5th_fold':
              num_test = 13154
            else:
              num_test = 13155
          elif Dataset == '2':
            num_test = 503
          elif Dataset == '3':
            num_test = 400
          print(num_test)

          for img_index in range(num_test):
            t_image = np.array(Test_Images[img_index,:]).reshape(1,784)
            t_label = np.array(Test_Labels[img_index,:]).reshape(1,number_of_classes)
            test_acc = accuracy.eval(feed_dict={
                x: t_image, y_: t_label,
                keep_prob: 1.0, training:False})
            Accuracy += test_acc
            wrong, predicted, prediction_prob = sess.run([wrong_prediction_printop, 
                                 predicted_labels_printop,prediction_prob_printop], 
                                feed_dict={
                x: t_image, y_: t_label, 
                keep_prob: 1.0, training:False})
            if img_index <= 3:
              b1, b2, b3, t1, t2, t3, t1_b, t2_b, t3_b = sess.run([b1_conv_printop, b2_conv_printop, b3_conv_printop,
                                  t1_conv_printop,t2_conv_printop, t3_conv_printop, t1_b_conv_printop, t2_b_conv_printop, t3_b_conv_printop], 
                                  feed_dict={
                  x: t_image, y_: t_label, 
                  keep_prob: 1.0, training:False})
              if img_index == 0:
                b1_list = b1
                b2_list = b2
                b3_list = b3
                t1_list = t1
                t2_list = t2
                t3_list = t3
                t1_b_list = t1_b
                t2_b_list = t2_b
                t3_b_list = t3_b
              else:
                b1_list = np.append(b1_list,b1,axis=0)
                b2_list = np.append(b2_list,b2,axis=0)
                b3_list = np.append(b3_list,b3,axis=0)
                t1_list = np.append(t1_list,t1,axis=0)
                t2_list = np.append(t2_list,t2,axis=0)
                t3_list = np.append(t3_list,t3,axis=0)
                t1_b_list = np.append(t1_b_list,t1_b,axis=0)
                t2_b_list = np.append(t2_b_list,t2_b,axis=0)
                t3_b_list = np.append(t3_b_list,t3_b,axis=0)     
            if img_index == 0 :
              wrong_list_1 = wrong
              predicted_list_1 = predicted
              prediction_prob_1 = prediction_prob
            else:
              wrong_list_1 = np.append(wrong_list_1,wrong,axis=0)
              predicted_list_1 = np.append(predicted_list_1,predicted,axis=0)
              prediction_prob_1 = np.append(prediction_prob_1, prediction_prob)


          Accuracy = Accuracy/num_test
          accuracy_list.append(Accuracy)
          list_of_predicted_list.append(predicted_list_1)
          print('Average test accuracy: %g' % Accuracy)
          epoch_around = math.ceil((itrt_index * batch_size) / total_trainImages)
          sio.savemat('D'+Dataset+'_'+file_name+'_'+str(epoch_around)+'ep_'+data_type+'_predicted_labels_list.mat', {'wrong_list':wrong_list_1, 'predicted_list': predicted_list_1, 'Target_labels':Target_labels,  
                                                                                                       'prediction_prob':prediction_prob, 'b1_list':b1_list, 'b2_list':b2_list, 'b3_list':b3_list, 't1_list':t1_list,
                                                                                                                    't2_list':t2_list, 't3_list':t3_list, 't1_b_list':t1_b_list, 't2_b_list':t2_b_list, 't3_b_list':t3_b_list})

          elapsed_time = time.time() - start_time
          print('Elapsed Time: %f secs' % elapsed_time)
          print('Batch Size & Iteration & Total Train Imgs : %d & %d & %d' % (batch_size, itrt_index, total_trainImages))   
          print('learning_rate : %g ' % learn_rate)
          print('1st conv FMaps : %d ' % fm1) 
          print('number of layers in dense block : %d ' % num_layers)  
          print('growth rate(k_fm) : %d ' % k_fm)
          print('filter size : %d ' % fs)
          print('bottleneck : %d' % bottleneck)
          print('dropout prob : %g ' % dropout_prob)
          print('data_type :', data_type)

          print('file_name :', file_name)

          print('FC nodes : %d' % fc_nodes[0])

          epoch_around = (itrt_index * batch_size) / total_trainImages
          print('Number of epochs : %f ' % epoch_around)

          # plot the cost
          plt.plot(np.squeeze(costs))
          plt.ylabel('cost')
          plt.xlabel('iterations (per tens)')
          plt.title("Learning rate =" + str(learn_rate))
          plt.show()

        if i % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x: batch[0], y_: batch[1], 
              keep_prob: 1.0, training:False})
          print('step %d, training accuracy %g' % (i, train_accuracy))
        _, loss  = sess.run([train_step, cross_entropy], 
                                                   feed_dict={x: batch[0], y_: batch[1], 
                                                              keep_prob: dropout_prob, training:True})

        iteration_cost = 0                       # Defines a cost related to an epoch
        num_minibatches = int(total_trainImages / batch_size) # number of minibatches of size minibatch_size in the train set        
        iteration_cost += loss / num_minibatches
        costs.append(iteration_cost)
        if i % 100 == 0:
          print ('Loss: ',loss)


      Accuracy = 0
      training_time = time.time() - start_time
      print('Training Time: %f secs' % training_time)


      if Dataset == '1':
        if file_name == '5th_fold':
          num_test = 13154
        else:
          num_test = 13155
      elif Dataset == '2':
        num_test = 503
      elif Dataset == '3':
        num_test = 400
      print(num_test)

      for img_index in range(num_test):
        t_image = np.array(Test_Images[img_index,:]).reshape(1,784)
        t_label = np.array(Test_Labels[img_index,:]).reshape(1,number_of_classes)
        test_acc = accuracy.eval(feed_dict={
            x: t_image, y_: t_label,
            keep_prob: 1.0, training:False})
        Accuracy += test_acc
        wrong, predicted = sess.run([wrong_prediction_printop, predicted_labels_printop],  feed_dict={
            x: t_image, y_: t_label, 
            keep_prob: 1.0, training:False})
        if img_index <= 3:
          b1, b2, b3, t1, t2, t3, t1_b, t2_b, t3_b = sess.run([b1_conv_printop, b2_conv_printop, b3_conv_printop,
                              t1_conv_printop,t2_conv_printop, t3_conv_printop, t1_b_conv_printop, t2_b_conv_printop, t3_b_conv_printop], 
                              feed_dict={
                x: t_image, y_: t_label, 
                keep_prob: 1.0, training:False})
          if img_index == 0:
            b1_list = b1
            b2_list = b2
            b3_list = b3
            t1_list = t1
            t2_list = t2
            t3_list = t3
            t1_b_list = t1_b
            t2_b_list = t2_b
            t3_b_list = t3_b
          else:
            b1_list = np.append(b1_list,b1,axis=0)
            b2_list = np.append(b2_list,b2,axis=0)
            b3_list = np.append(b3_list,b3,axis=0)
            t1_list = np.append(t1_list,t1,axis=0)
            t2_list = np.append(t2_list,t2,axis=0)
            t3_list = np.append(t3_list,t3,axis=0)
            t1_b_list = np.append(t1_b_list,t1_b,axis=0)
            t2_b_list = np.append(t2_b_list,t2_b,axis=0)
            t3_b_list = np.append(t3_b_list,t3_b,axis=0)  
        if img_index == 0 :
          wrong_list = wrong
          predicted_list = predicted
        else:
          wrong_list = np.append(wrong_list,wrong,axis=0)
          predicted_list = np.append(predicted_list,predicted,axis=0)


      Accuracy = Accuracy/num_test
      print('Average test accuracy: %g' % Accuracy)
      accuracy_list.append(Accuracy)
      list_of_predicted_list.append(predicted_list)

      elapsed_time = time.time() - start_time
      print('Elapsed Time: %f secs' % elapsed_time)
      print('Batch Size & Iteration & Total Train Imgs : %d & %d & %d' % (batch_size, itrt_index, total_trainImages))   
      print('learning_rate : %g ' % learn_rate)
      print('1st conv FMaps : %d ' % fm1) 
      print('number of layers in dense block : %d ' % num_layers)  
      print('growth rate(k_fm) : %d ' % k_fm)
      print('filter size : %d ' % fs)
      print('bottleneck : %d' % bottleneck)
      print('dropout prob : %g ' % dropout_prob)
      print('data_type :', data_type)

      print('file_name :', file_name)

      print('FC nodes : %d' % fc_nodes[0])

      epoch_around = math.ceil((iteration * batch_size) / total_trainImages)
      if epoch_around == 51:
        epoch_around = 50
      print('Number of epochs : %f ' % epoch_around)


      # plot the cost
      plt.plot(np.squeeze(costs))
      plt.ylabel('cost')
      plt.xlabel('iterations (per tens)')
      plt.title("Learning rate =" + str(learn_rate))
      plt.show()

    sio.savemat('D'+Dataset+'_'+file_name+'_'+str(epoch_around)+'ep_'+data_type+'_predicted_labels_list.mat', {'wrong_list':wrong_list, 'predicted_list': predicted_list, 'Target_labels':Target_labels, 'accuracy_list':accuracy_list, 'list_of_predicted_list':list_of_predicted_list, 'costs':costs, 'b1_list':b1_list, 'b2_list':b2_list, 'b3_list':b3_list, 't1_list':t1_list,
                                                                                                                    't2_list':t2_list, 't3_list':t3_list, 't1_b_list':t1_b_list, 't2_b_list':t2_b_list, 't3_b_list':t3_b_list})
    
    
class MyModel:
  num_layers = 4
  k_fm = 24
  fs = 3
  fm1 = 32
  bottleneck = 4
  dropout_prob = 0.8
  batch_size = [16]
  learn_rate = 0.001
  num_of_test = 40

model = MyModel()
  

densenet('1','org',1,model)
densenet('1','org',2,model)
densenet('1','org',3,model)
densenet('1','org',4,model)
densenet('1','org',5,model)

densenet('1','aug',1,model)
densenet('1','aug',2,model)
densenet('1','aug',3,model)
densenet('1','aug',4,model)
densenet('1','aug',5,model)

densenet('2','org',1,model)
densenet('2','org',2,model)
densenet('2','org',3,model)
densenet('2','org',4,model)
densenet('2','org',5,model)
  
densenet('2','aug',1,model)
densenet('2','aug',2,model)
densenet('2','aug',3,model)
densenet('2','aug',4,model)
densenet('2','aug',5,model)

densenet('3','org',1,model)
densenet('3','org',2,model)
densenet('3','org',3,model)
densenet('3','org',4,model)
densenet('3','org',5,model)

densenet('3','aug',1,model)
densenet('3','aug',2,model)
densenet('3','aug',3,model)
densenet('3','aug',4,model)
densenet('3','aug',5,model)
