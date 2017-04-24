'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
This is an example of defining a linear regression model and using
the training utilities in TensorFlow to fit the model to the data.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
import tensorflow as tf

# Set logging
tf.logging.set_verbosity(tf.logging.INFO)

def model(features, labels, mode):
    # Define the regression model subgraph
    W = tf.get_variable('W', [1], dtype=tf.float64)
    b = tf.get_variable('b', [1], dtype=tf.float64)
    y = W * features['x'] + b

    # Define the loss subgraph
    loss = tf.reduce_sum(tf.square(y - labels))

    # Define the training subgraph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    # Using group means that the given operations must complete
    # before advancing... maybe
    train = tf.group(optimizer.minimize(loss), 
            tf.assign_add(global_step, 1))

    # Return model function that can be hooked into the estimator
    # constructor
    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train
    )

# Define the data set
x = np.array([1, 2, 3, 4], dtype=np.float64)
y = np.array([0, -1, -2, -3], dtype=np.float64)
input_fn = tf.contrib.learn.io.numpy_input_fn(
        {"x": x}, y, batch_size=4, num_epochs=1000)

# Run the estimator that fits and evaluates the model
estimator = tf.contrib.learn.Estimator(model_fn=model)
estimator.fit(input_fn=input_fn, steps=1000)
# Print the loss and global steps
print(estimator.evaluate(input_fn=input_fn, steps=10))
