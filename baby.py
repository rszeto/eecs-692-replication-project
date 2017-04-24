import tensorflow as tf

# Create session that evaluates parts of the graph
sess = tf.Session()

#####################################################################
# First example
# Adds two constant numbers
#####################################################################

# Define some nodes
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2) # Equivalent: `node3 = node1 + node2`

# Prints the results
print(sess.run(node1)) # 3.0
print(sess.run(node2)) # 4.0
print(sess.run(node3)) # 7.0

#####################################################################
# Placeholder example
#####################################################################

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.add(a, b)
d = tf.multiply(3.0, c) # Equivalent: `d = 3 * c`

# Result of c with two different settings of a and b
print(sess.run(c, {a: 3, b: 4})) # 7.0
print(sess.run(c, {a: [1, 3], b: [2, 4]})) # [3. 7.]
# Result of d from setting (a, b) or (c)
print(sess.run(d, {a: 3, b: 4})) # 21.0
print(sess.run(d, {c: 6})) # 18.0

#####################################################################
# (Learnable) variable example
#####################################################################

# Define the line y = 3x - 3. The arguments here are the values that
# the variables are set to whenever the global initializer is run.
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# W and b are not initialized by default, so call initializer
init = tf.global_variables_initializer()
sess.run(init)

# Evaluate on multiple values of x
# [0. 0.30000001  0.60000002  0.90000004]
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

#####################################################################
# Loss and variable-setting example
#####################################################################

# Evaluate SSD loss of y = 3x - 3 on data
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) # 23.66

# Set what W and b should be
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
# Actually assign values in the session
sess.run([fixW, fixb])
# Report the new loss
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) # 0.0

#####################################################################
# Gradient descent example
#####################################################################

# Define gradient descent optimizer w/ learning rate 0.01
optimizer = tf.train.GradientDescentOptimizer(0.01)
# Define trainer node that optimizes on SSD loss
train = optimizer.minimize(loss)

# Reset W and b to wrong values
sess.run(init)
print(sess.run([W, b]))
# Run optimizer and print intermediate results
print('Optimizing')
for i in range(500):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    # Print every 20 iterations
    if i % 20 == 0:
        # Get values of W, b, and loss
        W_arr, b_arr = sess.run([W, b])
        loss_val = sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
        print('Iteration: %03d, Loss: %.5f (W=%.5f, b=%.5f)' % (i, loss_val, W_arr[0], b_arr[0]))

# Print final values of W and b
W_arr, b_arr = sess.run([W, b])
print('Final variable values: W=%.5f, b=%.5f' % (W_arr, b_arr))