import tensorflow as tf
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, BatchNormalization,
GlobalAveragePooling2D, MaxPool2D, Dropout, ReLU, AveragePooling2D)
from tensorflow.keras import Model

class EDenseNet(Model):
  def __init__(self, param, name):
    super(EDenseNet, self).__init__(name=name)
    self.number_of_db = 3
    self.fm_1st_layer = param.fm_1st_layer
    self.growth_rate = param.growth_rate
    self.db_layer = param.num_layers

    init_TruncatedNormal = tf.keras.initializers.TruncatedNormal(
        mean=0., stddev=0.1, seed= 1)
    self.first_conv_layer = Conv2D(self.fm_1st_layer, param.filter_size, 
                                   padding='same', input_shape=(28, 28, 1), 
                                   use_bias=False, 
                                   kernel_initializer=init_TruncatedNormal)
    
    # Dense block
    # db_layer_ops[i][j][k] i: dense block, j: layer, k: operation
    self.db_layer_ops = []
    for i in range(self.number_of_db):
      layer = []
      for i in range(param.num_layers):
        ops = []
        ops.append(BatchNormalization())
        ops.append(ReLU())
        ops.append(Conv2D(self.growth_rate, param.filter_size, padding='same', 
                          use_bias=False, 
                          kernel_initializer=init_TruncatedNormal))
        ops.append(Dropout(param.dropout_prob, seed=1))
        layer.append(ops)
      self.db_layer_ops.append(layer)

    
    trans1_fm = self.fm_1st_layer + (self.db_layer * self.growth_rate)
    trans2_fm = trans1_fm + (self.db_layer * self.growth_rate)
    trans3_fm = trans2_fm + (self.db_layer * self.growth_rate)
    self.trans_fm = [trans1_fm, trans2_fm, trans3_fm]

    # Transition layer
    # trans_layer_ops[i][j][k] i: transition layer, j: layer, k: operation
    self.trans_layer_ops = []
    for i in range(self.number_of_db):
      layer = []
      
      # 1 x 1 bottleneck layer ops
      ops = []
      ops.append(BatchNormalization())
      ops.append(ReLU())
      ops.append(Conv2D(param.bottleneck * param.growth_rate, 1, 
                       padding='same', use_bias=False, 
                        kernel_initializer=init_TruncatedNormal))
      ops.append(Dropout(param.dropout_prob, seed=1))
      layer.append(ops)
      
      # Conv layer ops
      ops = []
      ops.append(BatchNormalization())
      ops.append(ReLU())
      ops.append(Conv2D(self.trans_fm[i], param.filter_size, padding='same', 
                        use_bias=False, 
                        kernel_initializer=init_TruncatedNormal))
      ops.append(Dropout(param.dropout_prob, seed=1))
      layer.append(ops)
      
      # # Orginal transition layer
      # ops = []
      # ops.append(BatchNormalization())
      # ops.append(ReLU())
      # ops.append(Conv2D(self.trans_fm[i], 1, padding='same', 
      #                   use_bias=False, 
      #                   kernel_initializer=init_TruncatedNormal))
      # ops.append(Dropout(param.dropout_prob, seed=1))
      # layer.append(ops)

      # Pooling layer
      # Check if it is the last layer, add pooling layer accrodingly
      ops = []
      if self.number_of_db - i == 1:
        ops.append(GlobalAveragePooling2D())
      else :
        ops.append(MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'))
        # ops.append(AveragePooling2D(pool_size=(2, 2), strides=None, 
        #                            padding='valid'))
      layer.append(ops)
      self.trans_layer_ops.append(layer)

      # Last layer, dense layer
      self.dense_layer = Dense(param.num_classes, 
                               kernel_initializer=init_TruncatedNormal)

  def call(self, x):
    x = self.first_conv_layer(x)
    # num_fm = self.fm_1st_layer

    for i in range(self.number_of_db):
      # Dense block and transition layer
      for j in range(self.db_layer):
        # Dense block
        # Layer ops in dense block
        temp = self.db_layer_ops[i][j][0](x) # Batch_norm
        temp = self.db_layer_ops[i][j][1](temp) # Relu
        temp = self.db_layer_ops[i][j][2](temp) # Conv2D
        temp = self.db_layer_ops[i][j][3](temp) # Dropout
        x = tf.keras.layers.Concatenate(axis=3)([x, temp])

      # Transition layer
      # 1 x 1 bottleneck layer ops
      x = self.trans_layer_ops[i][0][0](x) # Batch_norm
      x = self.trans_layer_ops[i][0][1](x) # Relu
      x = self.trans_layer_ops[i][0][2](x) # Conv2D 1 x 1
      x = self.trans_layer_ops[i][0][3](x) # Dropout
      # Conv layer
      x = self.trans_layer_ops[i][1][0](x) # Batch_norm
      x = self.trans_layer_ops[i][1][1](x) # Relu
      x = self.trans_layer_ops[i][1][2](x) # Conv2D
      x = self.trans_layer_ops[i][1][3](x) # Dropout
      # Pooling layer
      x = self.trans_layer_ops[i][2][0](x)

      # # Original transition layer
      # # 1 x 1 conv layer ops
      # x = self.trans_layer_ops[i][0][0](x) # Batch_norm
      # x = self.trans_layer_ops[i][0][1](x) # Relu
      # x = self.trans_layer_ops[i][0][2](x) # Conv2D 1 x 1
      # x = self.trans_layer_ops[i][0][3](x) # Dropout
      # # Pooling layer
      # x = self.trans_layer_ops[i][1][0](x)

    return self.dense_layer(x)

  def model(self):
    x = tf.keras.layers.Input(shape=(28, 28, 1))
    return Model(inputs=[x], outputs=self.call(x))