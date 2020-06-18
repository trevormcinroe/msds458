# Week 2: The XOR Problem

## Introduction

## What's in this directory?
This directory contains a from-scratch implementation of:
* Dense neural network (1 hidden layer)
    * week2/nn.py
* ADAM optimizer
    * week2/optimizer.py
* Backpropagation
    * week2/nn.get_gradient() + week2/optimizer.update_weights()
* Cross entropy loss
    * week2/loss.py


## Using the algorithm
The NN class has been inplemented to allow for a customer input size, output size, and number of hidden units.
For our XOR problem, please use an input size of 2 and output size of 2. You can use whatever number of hidden
units you would like.

For ADAM, the generally recommended hyperparameter values are as follows:
* ![](https://latex.codecogs.com/gif.latex?%5Calpha) = 0.01
* ![](https://latex.codecogs.com/gif.latex?%5Cbeta_m) = 0.99
* ![](https://latex.codecogs.com/gif.latex?%5Cbeta_v) = 0.999
* ![](https://latex.codecogs.com/gif.latex?%5Cepsilon) = 0.00001  

The xor_problem.py script performs the training loop and outputs a graph of the loss function 
to ```./week2/media/xor_loss.png```

The below code shows the initialization of the NN class with an input vector of size 2, (![](\boldsymbol{x} \in \mathbb{R}^2)),
10 hidden units, and producing 2 outputs per example in the batch. This results in the softmax function producing a
predicted probability for each class for each example. Finally, the NN class is given the seed 111 for 
reproducability.

The second line initialized the ADAM optimizer with the above recommended hyperparameters.

The "for loop" controls the number of epochs. The .forward_pass() method of the NN class produces an output
matrix and the .softmax() method passes that matrix through the softmax function.

The next line computes the cross-entropy loss between the forward pass and the corresponding training labels.

The following line calculates the gradient of the neural network via a backward pass (explained more in depth
in the following section).

This gradient, along with the network's current weights, are passes to the ADAM optimizer and the weights are
updated.
```python
nn = NN(input_length=2, n_hidden_units=10, n_outputs=2, seed=111)
ADAM = ADAM(layer_dims=nn.layer_dims, alpha=0.01, beta_m=0.99, beta_v=0.999, epsilon=0.00001)

for _ in range(100):
    output = nn.forward_pass(input=train_samples)
    sm_output = nn.softmax(input=output)
    loss = cross_entropy_loss(y_pred=sm_output, y_actual=train_labels)
    grad = nn.get_gradient(input=train_samples, y_pred=sm_output, y_actual=train_labels)
    ADAM.update_weights(weights=nn.weights, gradient=grad)
```

### Derivation of the backprop algorithm
(Hopefully this renders well. Github doesn't support rendering LaTeX natively.)

You can find the code for this in the $.get_gradient()$ method of the NN class.
![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Chat%7By%7D%20%26%3D%20softmax%28ReLU%28%5Cboldsymbol%7Bx%7D%5Cboldsymbol%7BW%7D_%7B0%7D%20&plus;%20%5Cboldsymbol%7Bb%7D_%7B0%7D%29%5Cboldsymbol%7BW%7D_%7B1%7D%20&plus;%20%5Cboldsymbol%7Bb%7D_%7B1%7D%29%20%5C%5C%20a_0%20%26%3D%20ReLU%28%5Cboldsymbol%7Bx%7D%5Cboldsymbol%7BW%7D_%7B0%7D%20&plus;%20%5Cboldsymbol%7Bb%7D_%7B0%7D%29%5C%5C%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cboldsymbol%7Bb%7D_1%7D%20%26%3D%20%5Chat%7By%7D%20-%20y%20%5C%5C%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cboldsymbol%7BW%7D_1%7D%20%26%3D%20a_%7B0%7D%5E%7BT%7D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cboldsymbol%7Bb%7D_1%7D%5C%5C%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cboldsymbol%7Bb%7D_0%7D%20%26%3D%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cboldsymbol%7Bb%7D_1%7D%20%5Cboldsymbol%7BW%7D_%7B1%7D%5E%7BT%7D%5C%5C%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cboldsymbol%7BW%7D_0%7D%20%26%3D%20%5Cboldsymbol%7Bx%7D%5E%7BT%7D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cboldsymbol%7Bb%7D_0%7D%5C%5C%20%5Cend%7Balign*%7D)

You will notice that the derivative of the softmax function simplifies to ![](https://latex.codecogs.com/gif.latex?%5Chat%7By%7D%20-%20y).
This is one of the many nice properties of the softmax function. For the proof of this derivation, see:
https://www.ics.uci.edu/~pjsadows/notes.pdf (Page 3, eqn's 17-27).