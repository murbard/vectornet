Many gradient descent algorithms (Stochastic gradient descent, Nesterov acceleration, L-BFGS...) can be  described succintly as recursive neural networks with access to gradient information and operating on vector valued neurons and scalar valued neurons, with a dot product operation as well as operations in the vector space.

This project introduces vector valued neurons of symbolic length n, and a dot product in an attempt to learn efficient gradient descent algorithm for small problems that will generalize well when used on very large problems.
