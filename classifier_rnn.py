
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

import os
import glob
import random
from tensorflow.contrib import rnn



file_name_list = glob.glob('*txt')
 
dic = {}
dic = {'\n':1}
dic = {'<PAD>':0}
i=2;

openfile = [open(file_name) for file_name in file_name_list]
for f in openfile:

	all_words = map(lambda l: l.lower().split(" "), f.readlines())
	for index in range(len(all_words)):
		t = all_words[index];
		for ind in range(len(t)):
			m = t[ind];
			if m in dic:
				pass;
			else:
				dic[m] = i;
				i = i+1;

f.close()

# print(dic['in'])

train_neg_data = []
file = open("sem13_neg_text.txt", "r")
all_words = map(lambda l: l.lower().split(" "), file.readlines())
for index in range(len(all_words)):
	t = all_words[index];
	temp = [];
	for ind in range(len(t)):
		m = t[ind];
		if m in dic:
			temp.append(dic[m]);
		else:
			temp.append(0);
		
	train_neg_data.append(temp)
file.close()

# print(train_neg_data[1])

train_pos_data = []
file = open("sem13_pos_text.txt", "r")
all_words = map(lambda l: l.lower().split(" "), file.readlines())
for index in range(len(all_words)):
	t = all_words[index];
	temp = [];
	for ind in range(len(t)):
		m = t[ind];
		if m in dic:
			temp.append(dic[m]);
		else:
			temp.append(0);
	train_pos_data.append(temp)

file.close()

train_neu_data = []
file = open("sem13_neu_text.txt", "r")
all_words = map(lambda l: l.lower().split(" "), file.readlines())
for index in range(len(all_words)):
	t = all_words[index];
	temp = [];
	for ind in range(len(t)):
		m = t[ind];
		if m in dic:
			temp.append(dic[m]);
		else:
			temp.append(0);
	train_neu_data.append(temp)

file.close()

test_pos = [];
file = open("sem13_pos_test_text", "r")
all_words = map(lambda l: l.lower().split(" "), file.readlines())
for index in range(len(all_words)):
	t = all_words[index];
	temp = [];
	for ind in range(len(t)):
		m = t[ind];
		if m in dic:
			temp.append(dic[m]);
		else:
			temp.append(0);
	test_pos.append(temp)

file.close()

test_neg = [];
file = open("sem13_neg_test_text", "r")
all_words = map(lambda l: l.lower().split(" "), file.readlines())
for index in range(len(all_words)):
	t = all_words[index];
	temp = [];
	for ind in range(len(t)):
		m = t[ind];
		if m in dic:
			temp.append(dic[m]);
		else:
			temp.append(0);
	test_neg.append(temp)

file.close()

test_neu = [];
file = open("sem13_neu_test_text", "r")
all_words = map(lambda l: l.lower().split(" "), file.readlines())
for index in range(len(all_words)):
	t = all_words[index];
	temp = [];
	for ind in range(len(t)):
		m = t[ind];
		if m in dic:
			temp.append(dic[m]);
		else:
			temp.append(0);
	test_neu.append(temp)

file.close()




testing_data  = test_pos+test_neg+test_neu
testing_data = keras.preprocessing.sequence.pad_sequences(testing_data,value=dic["<PAD>"],padding='post',maxlen=50)
test_pos_len = len(test_pos)
test_neg_len = len(test_neg)
test_neu_len = len(test_neu)

testing_label = [[1,0,0]]*test_pos_len + [[0,1,0]]*test_neg_len+[[0,0,1]]*test_neu_len
testing_label = np.asarray(testing_label)




training_data=train_pos_data+train_neg_data+train_neu_data
training_data = keras.preprocessing.sequence.pad_sequences(training_data,value=dic["<PAD>"],padding='post',maxlen=50)
training_data = np.asarray(training_data)
# print(training_data[7])

vocab = len(dic)+1
print(vocab)
pos_len = len(train_pos_data)
neg_len = len(train_neg_data)
neu_len = len(train_neu_data)

training_label = [[1,0,0]]*pos_len+[[0,1,0]]*neg_len+[[0,0,1]]*neu_len




# training_label=np.asarray(training_label)
print(training_data.shape)

# print(training_data[0],"1st")
# c = list(zip(training_data, training_label))

# random.shuffle(c)

# training_data, training_label = zip(*c)
training_data = np.asarray(training_data)
training_label=np.asarray(training_label)
# print(training_data[0],"2nd")


print (training_data.shape,training_label.shape)


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

num_input = 50          # MNIST data input (image shape: 28x28)
timesteps = 50        # Timesteps
n_classes = 3

learning_rate = 0.001 # The optimization initial learning rate
epochs = 8           # Total number of training epochs
batch_size = 64     # Training batch size
display_freq = 50    # Frequency of displaying the training results

num_hidden_units = 64  # Number of hidden units of the RNN

def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.1)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initial)


def RNN(x, weights, biases,timesteps):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    # x = tf.unstack(x, timesteps, 1)
    #x = tf.reshape(x, [1,50])

    # # Generate a n_input-element sequence of inputs
    # # (eg. [had] [a] [general] -> [20] [6] [33])
    #x = tf.split(x,num_input,1)

    # 1-layer LSTM with n_hidden units.
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # Define a rnn cell with tensorflow
    rnn_cell = tf.contrib.rnn.BasicRNNCell(num_hidden_units)

    # Get lstm cell output
    # If no initial_state is provided, dtype must be specified
    # If no initial cell state is provided, they will be initialized to zero
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    #print(outputs.shape)
    # Linear activation, using rnn inner loop last output
    return tf.add(tf.matmul(outputs[:,-1], weights),biases)

x = tf.placeholder(tf.int32, shape=[None, timesteps], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')

# weights = {
#     'out': tf.Variable(tf.random_normal([num_hidden_units,3]))
# }
# biases = {
#     'out': tf.Variable(tf.random_normal([3]))
# }
# create weight matrix initialized randomely from N~(0, 0.01)
weights = weight_variable(shape=[num_hidden_units, n_classes])

# create bias vector initialized as zero
biases = bias_variable(shape=[n_classes])

word_embeddings = tf.get_variable("word_embeddings",[vocab,50])
embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, x)
print(embedded_word_ids.shape,'before')
output_logits = RNN(embedded_word_ids,weights,biases,timesteps)
y_pred = tf.nn.softmax(output_logits)

cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')



init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
global_step = 0
# Number of training iterations in each epoch
num_tr_iter = int(len(training_label) / batch_size)

for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    x_train, y_train = randomize(training_data,training_label)
    for iteration in range(num_tr_iter):
        global_step += 1
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
        # print(len(x_train)*len(x_train[0]))
        # print(batch_size*timesteps*num_input)
        #x_batch = x_batch.reshape((batch_size,timesteps, num_input))
        # Run optimization op (backprop)
        feed_dict_batch = {x: x_batch, y: y_batch}
        sess.run(optimizer, feed_dict=feed_dict_batch)

        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            # loss_batch, acc_batch = sess.run([loss, accuracy],
            #                                  feed_dict=feed_dict_batch)
            l = loss.eval(feed_dict= feed_dict_batch)
            acc = accuracy.eval(feed_dict = feed_dict_batch)

            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                  format(iteration, l, acc))



feed_dict_test = {x: testing_data, y: testing_label}
loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
print('---------------------------------------------------------')
print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
print('---------------------------------------------------------')


