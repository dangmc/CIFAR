import cPickle as pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.dtypes import float32
FLAGS = tf.app.flags.FLAGS

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation, :, :, :]
  shuffled_labels = labels[permutation, :]
  return shuffled_dataset, shuffled_labels


path = '/home/dangmc/Documents/DeepLearning/Data/cifar-10/'
image_size = 32
number_channels = 3
number_labels = 10;
train_size = 50000;
test_size = 10000;

def     readFile(filename):
    fo = open(path+filename, 'rb')
    dict = pickle.load(fo)
    fo.close()
    datasets = dict['data'];
    labels = np.array(dict['labels']).reshape(test_size);
    return datasets, labels

def	Reshape(datasets, labels):
  datasets = datasets.reshape((-1, number_channels, image_size*image_size))
  datasets = datasets.transpose([0, 2, 1])
  datasets = datasets.reshape((-1, image_size, image_size, number_channels)).astype(np.float32)
  labels = (np.arange(number_labels) == labels[:, None]).astype(np.int32)
  return datasets, labels

def make_train_datasets():
    train_dataset = np.ndarray(shape=(train_size, image_size, image_size, number_channels), dtype=np.float32)
    train_labels = np.ndarray(shape=(train_size, number_labels), dtype=np.int32);

    index = 0;
    for id in xrange(5):
        filename = "data_batch_" + str(id + 1);
        data, label = readFile(filename);

        data, label = Reshape(data, label)
        for i in xrange(len(label)):
            train_dataset[index, :, :, :] = data[i, :, :, :];
            train_labels[index, :] = label[i, :]
            index += 1
    train_dataset /= 255;
    return train_dataset[0:45200], train_labels[0:45200], train_dataset[45200:], train_labels[45200:];


def make_test_datasets():
    test_dataset = np.ndarray(shape=(test_size, image_size, image_size, number_channels), dtype=np.float32)
    test_labels = np.ndarray(shape=(test_size, number_labels), dtype=np.int32);
    filename = "test_batch";
    data, label = readFile(filename);
    data, label = Reshape(data, label)
    for i in xrange(len(label)):
        test_dataset[i, :, :, :] = data[i, :, :, :];
        test_labels[i, :] = label[i, :]
    test_dataset /= 255
    return test_dataset, test_labels

train_dataset, train_labels, validation_dataset, validation_labels = make_train_datasets()
test_dataset, test_labels = make_test_datasets();
batch_test_dataset = test_dataset[0:32]
batch_test_labels = test_labels[0:32]


batch_size = 32
patch_size = 5
depth = 64
depth_1 = 16
depth_2 = 20
hidden_unit_1 = 384;
hidden_unit_2 = 192;
dim = 8*8*64

graph = tf.Graph()
with graph.as_default():

  def _variable_on_cpu(name, const, shape):
    var = tf.Variable(tf.constant(const, shape=shape))
    return var

  def _variable_with_weight_decay(name, shape, stddev, wd):
      var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32))
      if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
      return var


  tf_train_datasets = tf.placeholder(float32, shape=(batch_size, image_size, image_size, number_channels))
  tf_train_labels = tf.placeholder(float32, shape=(batch_size, number_labels))
  tf_learning_rate = tf.placeholder(float32)
  tf_decay_steps = tf.placeholder(float32)
  tf_decay_rate = tf.placeholder(float32)
  tf_lamda = tf.placeholder(float32)

  tf_test_datasets = tf.placeholder(float32, shape=(batch_size, image_size, image_size, number_channels))



  weight_layer_1 = _variable_with_weight_decay('weights_1',
                                         shape=[1, patch_size, number_channels, depth],
                                         stddev=5e-2, wd=0.0)
  bias_layer_1 = _variable_on_cpu("bias_1", 0.0, [depth])

  weight_layer_2 = _variable_with_weight_decay('weights_2',
                                                 shape=[patch_size, 1, depth, depth],
                                                 stddev=5e-2, wd=0.0)
  bias_layer_2 = _variable_on_cpu("bias_2", 0.1, [depth])

  weight_layer_3 = _variable_with_weight_decay('weights_3',
                                         shape=[1, patch_size, depth, depth],
                                         stddev=5e-2, wd=0.0)
  bias_layer_3 = _variable_on_cpu("bias_3", 0.0, [depth])

  weight_layer_4 = _variable_with_weight_decay('weights_4',
                                                 shape=[patch_size, 1, depth, depth],
                                                 stddev=5e-2, wd=0.0)
  bias_layer_4 = _variable_on_cpu("bias_4", 0.1, [depth])

  weight_fully_1 = _variable_with_weight_decay('weights_fully_1', shape=[dim, hidden_unit_1],
                                         stddev=0.04, wd=0.004)
  biases_fully_1 = _variable_on_cpu('biases_fully_1', 0.1, [hidden_unit_1])

  weight_fully_2 = _variable_with_weight_decay('weights_fully_2', shape=[hidden_unit_1, hidden_unit_2],
                                          stddev=0.04, wd=0.004)
  biases_fully_2 = _variable_on_cpu('biases_fully_2', 0.1, [hidden_unit_2])

  weight_fully_3 = _variable_with_weight_decay('weights_fully_3', [192, number_labels],
                                          stddev=1/192.0, wd=0.0)
  biases_fully_3 = _variable_on_cpu('biases_fully_3', 0.0, [number_labels])
  def get_l2_loss(l2_lambda, layer_weights):
    return l2_lambda * sum(map(tf.nn.l2_loss, layer_weights))

  def model_train(data):
      conv_net = tf.nn.conv2d(data, weight_layer_1, [1, 1, 1, 1], padding="SAME")
      hidden_relu = tf.nn.relu(tf.nn.bias_add(conv_net, bias_layer_1))

      conv_net = tf.nn.conv2d(hidden_relu, weight_layer_2, [1, 1, 1, 1], padding="SAME")
      hidden_relu = tf.nn.relu(tf.nn.bias_add(conv_net, bias_layer_2))

      hidden_pooling = tf.nn.max_pool(hidden_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")
      norm_1 = tf.nn.lrn(hidden_pooling, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

      conv_net = tf.nn.conv2d(norm_1, weight_layer_3, [1, 1, 1, 1], padding="SAME")
      hidden_relu = tf.nn.relu(tf.nn.bias_add(conv_net, bias_layer_3))

      conv_net = tf.nn.conv2d(hidden_relu, weight_layer_4, [1, 1, 1, 1], padding="SAME")
      hidden_relu = tf.nn.relu(tf.nn.bias_add(conv_net, bias_layer_4))

      norm_2 = tf.nn.lrn(hidden_relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
      hidden_pooling = tf.nn.max_pool(norm_2, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")

      reshape = tf.reshape(hidden_pooling, [batch_size, -1])

      fully_layer_1 = tf.nn.relu(tf.matmul(reshape, weight_fully_1) + biases_fully_1)

      activation = tf.nn.dropout(fully_layer_1, 0.5)

      fully_layer_2 = tf.nn.relu(tf.matmul(activation, weight_fully_2) + biases_fully_2)

      activation = tf.nn.dropout(fully_layer_2, 0.5)

      softmax_linear = tf.add(tf.matmul(activation, weight_fully_3), biases_fully_3)

      return softmax_linear


  def model_test(data):
      conv_net = tf.nn.conv2d(data, weight_layer_1, [1, 1, 1, 1], padding="SAME")
      hidden_relu = tf.nn.relu(tf.nn.bias_add(conv_net, bias_layer_1))

      conv_net = tf.nn.conv2d(hidden_relu, weight_layer_2, [1, 1, 1, 1], padding="SAME")
      hidden_relu = tf.nn.relu(tf.nn.bias_add(conv_net, bias_layer_2))

      hidden_pooling = tf.nn.max_pool(hidden_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")
      norm_1 = tf.nn.lrn(hidden_pooling, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

      conv_net = tf.nn.conv2d(norm_1, weight_layer_3, [1, 1, 1, 1], padding="SAME")
      hidden_relu = tf.nn.relu(tf.nn.bias_add(conv_net, bias_layer_3))

      conv_net = tf.nn.conv2d(hidden_relu, weight_layer_4, [1, 1, 1, 1], padding="SAME")
      hidden_relu = tf.nn.relu(tf.nn.bias_add(conv_net, bias_layer_4))

      norm_2 = tf.nn.lrn(hidden_relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
      hidden_pooling = tf.nn.max_pool(norm_2, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")

      reshape = tf.reshape(hidden_pooling, [batch_size, -1])

      fully_layer_1 = tf.nn.relu(tf.matmul(reshape, weight_fully_1) + biases_fully_1)

      fully_layer_2 = tf.nn.relu(tf.matmul(fully_layer_1, weight_fully_2) + biases_fully_2)

      softmax_linear = tf.add(tf.matmul(fully_layer_2, weight_fully_3), biases_fully_3)

      return softmax_linear



  logits = model_train(tf_train_datasets)

  layer_weights = [weight_fully_1, weight_fully_2]

  cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) \
                  + get_l2_loss(tf_lamda, layer_weights);

  global_step = tf.Variable(0, name="global_step", trainable=False)


  decayed_learning_rate = tf.train.exponential_decay(tf_learning_rate, global_step, tf_decay_steps, tf_decay_rate)
  optimizer = tf.train.GradientDescentOptimizer(decayed_learning_rate).minimize(cost_function, global_step=global_step)


  train_prediction = tf.nn.softmax(logits)


  test_prediction = tf.nn.softmax(model_test(tf_test_datasets))


def   accuracy(predictions, labels):
  return 100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

num_epochs = 50
num_steps = train_labels.shape[0] / batch_size
num_steps_validation = validation_labels.shape[0] / batch_size
print 'training size = %d' % (train_labels.shape[0])
print 'num_steps_validation = %d' % (num_steps_validation)
print 'validation size = %d' % (validation_labels.shape[0])

acccur = []
loss = []
with tf.Session(graph=graph) as session:

  tf.initialize_all_variables().run()
  print ("Initialized")

  for epoch in xrange(num_epochs):
    print 'epoch = %d' % (epoch)
    train_dataset, train_labels = randomize(train_dataset, train_labels)
    for step in xrange(num_steps):
        offset = step * batch_size
        batch_data = train_dataset[offset : (offset + batch_size), :, :, :]
        batch_labels = train_labels[offset : (offset + batch_size), :]

        feed_dict = {tf_train_datasets : batch_data, tf_train_labels : batch_labels, tf_learning_rate : 0.05, tf_decay_steps : num_steps * 10,
                     tf_decay_rate : 0.1, tf_lamda : 0.004}

        _, costFunction, predictions = session.run([optimizer, cost_function, train_prediction], feed_dict=feed_dict)


        if (step == num_steps - 1):
          print('Loss at epoch %d: %f' % (epoch, costFunction))
          percison = accuracy(predictions, batch_labels);
          print('Training accuracy: %.2f%%' % (percison))
          loss.append(percison)


    sum = 0
    for iter in xrange(num_steps_validation):
        offset = iter * batch_size
        batch_data = validation_dataset[offset : (offset + batch_size), :, :, :]
        batch_labels = validation_labels[offset : (offset + batch_size), :]
        predictions = test_prediction.eval({tf_test_datasets : batch_data})
        sum += np.sum(np.argmax(predictions, 1) == np.argmax(batch_labels, 1))
    print sum
    print('Validation accuracy: %.2f%%' % (100 * sum / validation_labels.shape[0]))
    acccur.append(100 * sum / validation_labels.shape[0])
    if (epoch % 6 == 0):
        sum = 0
        for iter in xrange(312):
            offset = iter * batch_size
            batch_data = test_dataset[offset : (offset + batch_size), :, :, :]
            batch_labels = test_labels[offset : (offset + batch_size), :]
            predictions = test_prediction.eval({tf_test_datasets : batch_data})
            sum += np.sum(np.argmax(predictions, 1) == np.argmax(batch_labels, 1))
        print('Test accuracy: %.2f%%' % (100 * sum / test_labels.shape[0]))

pickle_file = 'CNN_v1.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'accuracy': acccur,
    'loss' : loss
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

plt.plot(xrange(num_epochs), acccur, marker='.', label='Accuracy Validation');
#plt.plot(range(n_epochs), test_errors, marker='.', label='Test Data');
plt.xlabel('Number of epochs');
plt.ylabel('accuracy (%)');
plt.legend()
plt.grid()
plt.show()
