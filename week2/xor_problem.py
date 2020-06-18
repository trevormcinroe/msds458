"""
This is the main, runnable .py script for the xor-problem
"""

from week2.nn import NN
from week2.optimizer import ADAM
from week2.loss import cross_entropy_loss
import numpy as np
import matplotlib.pyplot as plt

train_samples = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

train_labels = np.array([0, 1, 1, 0])

# Initializing our neural networl and ADAM classes
nn = NN(input_length=2, n_hidden_units=10, n_outputs=2, seed=111)
ADAM = ADAM(layer_dims=nn.layer_dims, alpha=0.01, beta_m=0.99, beta_v=0.999, epsilon=0.00001)

# Running our training loop for 100 epochs with the entirety of our training data at each batch
# We'll also be keeping track of our loss at each step...
historical_losses = list()

for _ in range(100):
    output = nn.forward_pass(input=train_samples)
    sm_output = nn.softmax(input=output)
    loss = cross_entropy_loss(y_pred=sm_output, y_actual=train_labels)
    grad = nn.get_gradient(input=train_samples, y_pred=sm_output, y_actual=train_labels)
    ADAM.update_weights(weights=nn.weights, gradient=grad)
    historical_losses.append(loss)

# Our final prediction...
y_pred = nn.softmax(nn.forward_pass(input=train_samples))
print(np.argmax(y_pred, axis=1))

# Converting the historical_loss list into a plot...
plt.plot(historical_losses)
plt.xlabel('Epoch')
plt.ylabel('Cross-entropy loss')
plt.title('Loss per training epoch')
plt.savefig('./media/xor_loss.png', dpi=400)