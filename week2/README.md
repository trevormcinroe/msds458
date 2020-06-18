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
* $`\beta_m`$ = 0.99
* $`\beta_v`$ = 0.999
* $`\epsilon`$ = 0.00001  
* $`a^2+b^2=c^2`$

The xor_problem.py script performs the training loop and outputs a graph of the loss function 
to ```./week2/media/xor_loss.png```

### Derivation of the backprop algorithm
```math
\hat{y} = softmax(ReLU(xW_0 + b_0)W_1 + b_1)\\
\delta_1 = \hat{y}
```
https://www.ics.uci.edu/~pjsadows/notes.pdf (Page 3, eqn's 17-27)