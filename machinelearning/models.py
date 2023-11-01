import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(batch_size=1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    converged = False
                    self.w.update(x, nn.as_scalar(y))


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        """
        成功的超参数：hidden_layer = 150, learning_rate = 0.005, batch_size = 100
        hidden_layer = 200, learning_rate = 0.01, batch_size = 200
        """
        self.hidden_layer = 200
        self.learning_rate = 0.01
        self.batch_size = 200
        self.w1 = nn.Parameter(1, self.hidden_layer)  # 因为特征只有一个，所以w1是1*hidden_layer的矩阵
        self.b1 = nn.Parameter(1, self.hidden_layer)  # b通常是1*hidden_layer的矩阵，因为每个样本处理后都会被加上相同的b
        self.w2 = nn.Parameter(self.hidden_layer, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        output1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        final_output = nn.AddBias(nn.Linear(output1, self.w2), self.b2)
        return final_output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        loss = 1

        while loss > 0.001:
            for x, y in dataset.iterate_once(self.batch_size):
                grad_w1, grad_b1, grad_w2, grad_b2 = nn.gradients(self.get_loss(x, y),
                                                                  [self.w1, self.b1, self.w2, self.b2])
                # nn.gradients 方法传入的参数需要按照计算时的顺序传入，否则会报错
                self.w1.update(grad_w1, -self.learning_rate)
                self.b1.update(grad_b1, -self.learning_rate)
                self.w2.update(grad_w2, -self.learning_rate)
                self.b2.update(grad_b2, -self.learning_rate)
                loss = nn.as_scalar(self.get_loss(x, y))
                # print(loss)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        # 二层神经网络似乎收敛速度较慢
        """
        layer1= 50, layer2 = 100, learning_rate = 0.1, batch_size = 500 epochs = 5的时候，准确率为0.92
        layer1= 100, layer2 = 100, learning_rate = 0.1, batch_size = 500 epochs = 5的时候，准确率为0.93
        layer1= 150, layer2 = 100, learning_rate = 0.1, batch_size = 500 epochs = 5的时候，准确率为0.93
        layer1= 200, layer2 = 100, learning_rate = 0.1, batch_size = 500 epochs = 5的时候，准确率为0.935
        layer1= 250, layer2 = 100, learning_rate = 0.1, batch_size = 500 epochs = 5的时候，准确率为0.935
        layer1= 300, layer2 = 100, learning_rate = 0.1, batch_size = 500 epochs = 5的时候，准确率为0.939
        layer1= 350, layer2 = 100, learning_rate = 0.1, batch_size = 500 epochs = 5的时候，准确率为0.937
        layer1= 400, layer2 = 100, learning_rate = 0.1, batch_size = 500 epochs = 5的时候，准确率为0.938
        layer1= 300, layer2 = 10, learning_rate = 0.1, batch_size = 500 epochs = 5的时候，准确率为0.932
        layer1= 300, layer2 = 50, learning_rate = 0.1, batch_size = 500 epochs = 5的时候，准确率为0.937
        layer1= 300, layer2 = 150, learning_rate = 0.1, batch_size = 500 epochs = 5的时候，准确率为0.937
        成功的超参数：layer1= 300, layer2 = 100, learning_rate = 0.5, batch_size = 250
        """
        self.hidden_layer1 = 300
        self.hidden_layer2 = 100
        self.learning_rate = 0.5  # learning_rate是最重要的参数，这个也和batch_size有关，而且不是越小越好，太小了会导致收敛速度过慢甚至不一定达到最优解
        self.batch_size = 250
        self.w1 = nn.Parameter(784, self.hidden_layer1)
        self.b1 = nn.Parameter(1, self.hidden_layer1)
        self.w2 = nn.Parameter(self.hidden_layer1, self.hidden_layer2)
        self.b2 = nn.Parameter(1, self.hidden_layer2)
        self.w_final = nn.Parameter(self.hidden_layer2, 10)
        self.b_final = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """

        output1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        output2 = nn.ReLU(nn.AddBias(nn.Linear(output1, self.w2), self.b2))
        final_output = nn.AddBias(nn.Linear(output2, self.w_final), self.b_final)
        return final_output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                grad_w1, grad_b1, grad_w2, grad_b2, grad_w_final, grad_b_final = nn.gradients(self.get_loss(x, y),
                                                                                              [self.w1, self.b1,
                                                                                               self.w2, self.b2,
                                                                                               self.w_final,
                                                                                               self.b_final])
                self.w1.update(grad_w1, -self.learning_rate)
                self.b1.update(grad_b1, -self.learning_rate)
                self.w2.update(grad_w2, -self.learning_rate)
                self.b2.update(grad_b2, -self.learning_rate)
                self.w_final.update(grad_w_final, -self.learning_rate)
                self.b_final.update(grad_b_final, -self.learning_rate)
                accuracy = dataset.get_validation_accuracy()
                # print(accuracy, accuracy > 0.975)
                if accuracy > 0.985:
                    return
                elif accuracy > 0.965:
                    self.learning_rate = 0.1


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
