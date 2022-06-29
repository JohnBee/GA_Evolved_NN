# Genetic Algorithm Evolved Neural Network
Using Genetic Algorithms to Evolve a Neural Network.

## About
Rather than using Backpropagation to update/optimise the weights of a 
Neural Network, here a Genetic Algorithm optimises the weights. Optimising Neural Networks usually
requires a training set, instead here using a fitness function. 

Within the Code an example of attempting to produce [10, -10] using an input of 0.1 is given.
The Network structure is [1, 10, 10, 2]. Input layer of size one, Two hidden layers of ten and
an output layer of size two.

The evaluation function used is a simple sum of absolute error between the target and the individual's output.

Using an initial population of 50, over 100 generations the final weights easily approximate the target as:
```
[10.001538363798826, -10.004689894474023]
```

## How to run the example code
```commandline
python genetic_algorithm.py 
```

These values can easily be adapted to Classification and other Regression problems, by changing the activation function within
the Neural Network definition.

## Requirements
 - Python >= 3.6
 - NumPy
 - DEAP

## Future Work
Use this same method to optimise an Ai to play a simple game. A Convolution Neural Network will read in the pixel values and
the Neural Network will output the next move.