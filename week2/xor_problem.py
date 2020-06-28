"""
This is the main, runnable .py script for the xor-problem
"""

from nn import NN
from optimizer import ADAM
from loss import cross_entropy_loss
from batching import MiniBatcher
import numpy as np
import matplotlib.pyplot as plt
import os

savefig_location = os.path.join(os.path.dirname(__file__), 'media')

train_samples = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

train_labels = np.array([0, 1, 1, 0])

# Initializing our neural networl and ADAM classes
nn = NN(input_length=2, n_hidden_units=10, n_outputs=2, seed=1111)
adam = ADAM(layer_dims=nn.layer_dims, alpha=0.01, beta_m=0.99, beta_v=0.999, epsilon=0.00001)
mb = MiniBatcher(data=train_samples, labels=train_labels, batch_size=4, seed=111)

# Running our training loop for 300 epochs with the entirety of our training data at each batch
# We'll also be keeping track of our loss at each step...
historical_losses = list()

EPOCHS = 300
epoch_counter = 0

while epoch_counter < EPOCHS:
    # Grabbing a mini-batch
    X_mb, y_mb = mb.fetch_minibatch()

    # Explicit check to see if we have run out of data
    # If so, increment the epoch and reset the MiniBatcher
    if isinstance(X_mb, bool):
        epoch_counter += 1
        mb.new_epoch()
        X_mb, y_mb = mb.fetch_minibatch()

    output = nn.forward_pass(input=X_mb)
    sm_output = nn.softmax(input=output)
    loss = cross_entropy_loss(y_pred=sm_output, y_actual=y_mb)
    grad = nn.get_gradient(input=X_mb, y_pred=sm_output, y_actual=y_mb)
    adam.update_weights(weights=nn.weights, gradient=grad)
    historical_losses.append(loss)

# Our final prediction...
y_pred = nn.softmax(nn.forward_pass(input=train_samples))
print(f'Trained network predictions: {np.argmax(y_pred, axis=1)}')
print(f'Ground-truth values: {train_labels}')
if np.array_equal(np.argmax(y_pred, axis=1), train_labels):
    print('Congrats, your network has solved the XOR problem!')
else:
    print('Looks like your network is not quite there... Try more epochs.')

# Converting the historical_loss list into a plot...
plt.plot(historical_losses)
plt.xlabel('Epoch')
plt.ylabel('Cross-entropy loss')
plt.title('Loss per training epoch')
plt.savefig(savefig_location + '/xor_loss.png', dpi=400)
