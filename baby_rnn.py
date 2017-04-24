import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell

# Create session that evaluates parts of the graph
sess = tf.Session()

# Define hyperparameters
num_hidden_units = 100
batch_size = 7
max_seq_len = 10
input_size = 4
output_size = 4
nonlinear_fn = tf.tanh

# Training data placeholders
data = tf.placeholder(tf.float32, [batch_size, max_seq_len, input_size])
labels = tf.placeholder(tf.float32, [batch_size, max_seq_len, output_size])

# Sequence of RNN states
cell = BasicRNNCell(num_hidden_units, activation=nonlinear_fn)
rnn_state, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

# Pass RNN state through FC layer. Need to reshape so FC layer is shared for all time steps
fc_weights = tf.Variable(tf.truncated_normal([num_hidden_units, output_size], stddev=0.1))
fc_bias = tf.Variable(tf.truncated_normal([output_size], stddev=0.1))
rnn_state_a = tf.reshape(rnn_state, [-1, num_hidden_units])
rnn_output = tf.matmul(rnn_state_a, fc_weights) + fc_bias
# Reshape to correct shape
unnorm_pred = tf.reshape(rnn_output, [-1, max_seq_len, output_size])

# Softmax
pred = tf.nn.softmax(unnorm_pred)
# Loss
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=unnorm_pred))
# Accuracy
acc_a = tf.equal(tf.argmax(pred, axis=2), tf.argmax(labels, axis=2))
acc = tf.reduce_mean(tf.cast(acc_a, "float"))

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy_loss)

# Define data (just a bunch of 1s)
batch_data = np.ones((batch_size, max_seq_len, input_size))
# Construct labels (a cycle through the possible outputs)
label = []
for i in range(max_seq_len):
    temp = np.zeros(output_size)
    temp[i % output_size] = 1
    label.append(temp)
repeat_label = [np.array(label) for _ in range(batch_size)]
batch_label = np.stack(repeat_label, axis=0)
train_dict = {data: batch_data, labels: batch_label}

# Initialize RNN weights
init = tf.global_variables_initializer()
sess.run(init)
# Optimize
for i in range(1000):
    sess.run(train, train_dict)
    # Print every 20 iterations
    if i % 100 == 0:
        loss_value, acc_value = sess.run([cross_entropy_loss, acc], train_dict)
        print('Iteration: %03d, Loss: %.5f, Acc: %.5f' % (i, loss_value, acc_value))

# Verify that predictions come from learned weights and bias
final_rnn_state, final_fc_weights, final_fc_bias, final_unnorm_pred = sess.run([rnn_state, fc_weights, fc_bias, unnorm_pred], train_dict)
for i in range(batch_size):
    for j in range(max_seq_len):
        cur_rnn_state = final_rnn_state[i, j, :]
        cur_unnorm_pred = np.dot(final_fc_weights.T, cur_rnn_state) + final_fc_bias
        assert(np.linalg.norm(cur_unnorm_pred - final_unnorm_pred[i, j, :]) < 0.00001)

# Get the predicted sequences
final_pred = sess.run(pred, train_dict)
for i in range(batch_size):
    print(final_pred[i, :, :])