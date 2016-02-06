__author__ = 'Arthur Breitman'
import ms_versions
import theano
import theano.tensor as T
import numpy as np

# initialize the pseudo random generator seed
rng = np.random.RandomState(1234)

n_batch = 20

class DotProducts(object):
    """
    A layer that computes the dot products of several linear combinations of vectors
    @param name: name of the layer, for debugging purposes
    @param n_input: number of input vector (NOT the dimension of the vector)
    @param n_output: number of output dot products
    @param w1_init: initial value for W1, which transforms the input vectors into n_output linear row vectors
    @param w2_init: initial value for W2, which creates n_ouput linear column vectors
    """

    def __init__(self, name, n_input, n_output, w1_init=None, w2_init=None):
        w_bound = np.sqrt(6.0 / (n_input + n_output))

        self.W1_init = (
            w1_init if w1_init else
            np.asarray(rng.uniform(low=-w_bound, high=w_bound, size=(n_input, n_output)), dtype=theano.config.floatX))

        self.W2_init = (
            w2_init if w2_init else
            np.asarray(rng.uniform(low=-w_bound, high=w_bound, size=(n_input, n_output)), dtype=theano.config.floatX))

        self.W1 = theano.shared(value=self.W1_init, name='%s.W1' % name, borrow=True)
        self.W2 = theano.shared(value=self.W2_init, name='%s.W2' % name, borrow=True)
        self.params = {self.W1, self.W2}

    def output(self, input_vectors):
        """
        Calculate the n_output dot product scalars of this layer
        @param input_vectors: n_input vectors (actual shape should be (n_batch, n_input, n_dimension)
        """

        return T.sum(T.tensordot(input_vectors, self.W1, [[1], [0]]) *
                     T.tensordot(input_vectors, self.W2, [[1], [0]]), axis=1)


class VectorCombination(object):
    """
    A fully connected layer building linear combinations of vectors with a non linearity
    @param name: name of the layer, for debugging purposes
    @param n_input: number of input vectors
    @param n_output: number of output vectors
    @param b_init: initial value for the bias
    """

    def __init__(self, name, n_input, n_output, b_init=None, activation='linear'):
        self.b_init = (
            b_init if b_init else
            np.zeros(dtype=theano.config.floatX, shape=(1, n_output, 1))
        )
        self.b = theano.shared(value=self.b_init, name='%s.b' % name, borrow=True)
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation
        self.params = {self.b}

    def output(self, input_vectors, input_scalars):
        """
        Calculate the n_output transformed vectors for this layer
        @param input_scalars: n_input x n_output scalar vector
        @param input_vectors: n_input vectors (actual shape should be (n_batch, n_input, n_dimension)
        """
        mat = input_scalars.reshape((n_batch, self.n_input, self.n_output))
        z = T.batched_tensordot(input_vectors, mat, [[1], [1]]).swapaxes(1, 2) + T.addbroadcast(self.b, 0, 2)
        if self.activation == 'linear':
            return z
        elif self.activation == 'rectified':
            return T.maximum(z, 0)
        elif self.activation == 'tanh':
            return T.tanh(z)
        else:
            raise "Unknown activation, %s" % self.activation


class LinearVectorCombination(object):
    def __init__(self, name, n_input, n_output, w_init=None, b_init=None):
        w_bound = np.sqrt(6.0 / (n_input + n_output))
        self.W_init = (
            w_init if w_init else
            np.asarray(rng.uniform(low=-w_bound, high=w_bound, size=(n_input, n_output)), dtype=theano.config.floatX))
        self.W = theano.shared(value=self.W_init, name='%s.W' % name, borrow=True)

        self.b_init = (
            b_init if b_init else
            np.zeros(dtype=theano.config.floatX, shape=(1, n_output, 1))
        )
        self.b = theano.shared(value=self.b_init, name='%s.b' % name, borrow=True)
        self.params = {self.W, self.b}

    def output(self, input_vectors):
        """
        Calculate the n_output transformed vectors for this layer
        @param input_vectors: n_input vectors (actual shape should be (n_batch, n_input, n_dimension)
        """
        return T.tensordot(input_vectors, self.W, [[1], [0]]).swapaxes(1, 2) + T.addbroadcast(self.b, 0, 2)


class ScalarLayer(object):
    """
    A fully connected layer from scalar to scalar, with non linearity
    @param name: name of the layer, for debugging purposes
    @param n_input: number of scalars in the input
    @param n_output: number of scalars produced
    @param w_init: initial value for the transform matrix
    @param b_init: initial value for the bias
    """

    def __init__(self, name, n_input, n_output, w_init=None, b_init=None, activation='tanh'):
        w_bound = np.sqrt(6.0 / (n_input + n_output))
        self.activation = activation
        self.W_init = (
            w_init if w_init else
            np.asarray(rng.uniform(low=-w_bound, high=w_bound, size=(n_input, n_output)), dtype=theano.config.floatX))
        self.W = theano.shared(value=self.W_init, name='%s.W' % name, borrow=True)

        self.b_init = (
            b_init if b_init else
            np.zeros(dtype=theano.config.floatX, shape=(1, n_output))
        )
        self.b = theano.shared(value=self.b_init, name='%s.b' % name, borrow=True)

        self.params = {self.W, self.b}

    def output(self, input_scalars):
        """
        Computes the n_output output scalars
        @param input_scalars: the layer's input
        @return: n_output scalars
        """
        z = T.dot(input_scalars, self.W) + T.addbroadcast(self.b, 0)
        if self.activation == 'linear':
            return z
        elif self.activation == 'rectified':
            return T.maximum(z, 0)
        elif self.activation == 'tanh':
            return T.tanh(z)
        else:
            raise "Invalid activation %s" % self.activation


class FunctionNode(object):
    """
    Computes the gradient with respect to an objective function
    @param name: name of node, useful for debugging purposes
    @param f: the function to take the gradient of
    """

    def __init__(self, name, f):
        self.name = name
        self.f = f

    def output(self, x):
        """
        Compute the output, the score and gradient at point x
        @param x: input vector
        @return: the vector gradient at point x
        """
        return self.f(x), T.grad(T.sum(self.f(x)), x)


class Unit(object):
    """
    A single unit of the recurrent neural network
    @param name: name of the unit, for debugging purposes
    @param f_node: the function we seek to minimize
    """

    def __init__(self, name, f_node,
                 n_dot_products=50,
                 n_scalars=((100, 200, 200, 200), (150, 200, 200, 99)),
                 n_vectors=(10, 20, 10)):
        self.name = name
        self.f_node = f_node
        self.n_dot_products = n_dot_products
        self.n_scalars = n_scalars
        self.n_vectors = n_vectors
        self.n_scalar_layers = [len(x)-1 for x in n_scalars]
        self.params = set()

        self.scalar_layers_A = [
            ScalarLayer("%s.ScalarLayer.A.%d" % (name, i), n_scalars[0][i], n_scalars[0][i + 1])
            for i in range(0, self.n_scalar_layers[0])]
        self.params.update(*[layer.params for layer in self.scalar_layers_A])

        self.scalar_layers_B = [
            ScalarLayer("%s.ScalarLayer.B.%d" % (name, i), n_scalars[1][i], n_scalars[1][i + 1])
            for i in range(0, self.n_scalar_layers[1])]
        self.params.update(*[layer.params for layer in self.scalar_layers_B])

        self.vector_combination = VectorCombination(
            "%s.VectorCombination" % name, n_vectors[0], n_vectors[1], activation='tanh')
        self.params.update(self.vector_combination.params)

        self.linear_vector_combination = LinearVectorCombination(
            "%s.LinearVectorCombination" % name, n_vectors[1], n_vectors[2])
        self.params.update(self.linear_vector_combination.params)

        self.dot_products = DotProducts("%s.DotProducts" % self.name, self.n_vectors[1], self.n_dot_products)
        self.params.update(self.dot_products.params)

    def output(self, x, hidden_vectors, hidden_scalars):
        """
        Computes the
        @param x: the current candidate point
        @param hidden_vectors: hidden vector memory
        @param hidden_scalars: hidden scalar memory
        @return: new candidate position, modified hidden vectors and modified hidden scalars
        """
        y, g = self.f_node.output(x)  # first capture the value of the function and

        scalars = T.concatenate([hidden_scalars, y.reshape((y.shape[0], 1))], axis=1)
        vectors = T.concatenate([hidden_vectors, g.reshape((g.shape[0], 1, g.shape[1]))], axis=1)

        s0 = reduce(lambda s, layer: layer.output(s), self.scalar_layers_A, scalars)

        vectors = self.vector_combination.output(vectors, s0)
        dp = self.dot_products.output(vectors)

        vectors = self.linear_vector_combination.output(vectors)

        s1 = T.concatenate([scalars, dp], axis=1)
        s1 = reduce(lambda s, layer: layer.output(s), self.scalar_layers_B, s1)

        return vectors[:, 0], vectors[:, 1:], s1


def objective(x):
    """
    objective function
    @param x: input vector
    @return: value of objective function
    """
    z = x - objective.offset
    return T.sum(T.pow(z, 4) - 16 * T.pow(z, 2) + 5 * z, axis=1) / 2



class SimpleClassifier(object):

    def __init__(self, mini_batch, classes, layer_shapes):

        self.mini_batch = mini_batch
        self.classes = classes
        self.layer_shapes = layer_shapes
        self.n_params = sum([np.prod(shape)+shape[1] for shape in layer_shapes])


    def objective(self, x):
        # first, reshape x into a set of parameters we need
        i, W, b = 0, [], []
        for shape in self.layer_shapes:
            l = np.prod(shape)
            W.append(x[:, i:i+l].reshape((n_batch,)+shape))
            i += l
            l = shape[1]
            b.append(x[:, i:i+l].reshape((n_batch, 1, l)))
        # calculate the cost
        z = T.tile(self.mini_batch.reshape((1, 50, 784)), (20, 1, 1))
        for wi, bi in zip(W, b):
            z = T.nnet.sigmoid(T.batched_dot(z, wi) + T.addbroadcast(bi,1))
        return T.mean((z-T.addbroadcast(T.extra_ops.to_one_hot(self.classes, 10).reshape((1, 50, 10)),0))**2, axis=2)



def run_main():
    """
    run the functions we want for main
    mainly there to avoid linter complaining about scope shadowing
    """

    images = np.load('/var/amm/breitart/mnist_train_images.npy').reshape(-1, 28*28)
    labels = np.load('/var/amm/breitart/mnist_train_labels.npy')

    choice = np.random.choice(len(images), 50)
    mini_batch_array = images[choice]
    classes_array = labels[choice].astype(int)

    mini_batch = theano.shared(name='mini_batch_input', value=mini_batch_array, borrow=True)
    classes = theano.shared(name='mini_labels', value=classes_array, borrow=True)

    classifier = SimpleClassifier(mini_batch, classes, [(28*28, 392), (392, 196), (196, 98), (98, 10)])

    fn = FunctionNode('objective', classifier.objective)

    u0 = Unit("gdu0", fn)
    ux = Unit("gdux", fn)

    local_x0 = np.random.randn(n_batch,  classifier.n_params)

    x0 = theano.shared(name='x0', value=local_x0, borrow=True)
    v0 = theano.shared(name='v0', value=np.zeros(shape=(n_batch, 9, classifier.n_params)))
    s0 = theano.shared(name='s0', value=np.zeros(shape=(n_batch, 99)))

    x, v, s = u0.output(x0, v0, s0)
    for i in range(0, 3):
        x, v, s = ux.output(x, v, s)

    y = classifier.objective(x)
    params = list(set.union(u0.params, ux.params))
    grad = T.grad(T.mean(y), params)
    updates = [(p, p - 0.01 * T.clip(g, -1, 1)) for (p, g) in zip(params, grad)]
    train = theano.function([], y, updates=updates, name='train')

    print "training..."
    for i in range(0, 50000):
        tt = train()
        if i % 1 == 0:
            print i, np.mean(tt)
        # change starting point
        np.copyto(local_x0, np.random.randn(n_batch,  classifier.n_params))
        # change objective to avoid memorizing the answer

        choice = np.random.choice(len(images), 50)
        np.coptyo(mini_batch_array, images[choice])
        np.copyto(classes_array, labels[choice])


if __name__ == '__main__':
    run_main()
    print "done."
