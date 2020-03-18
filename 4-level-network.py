import numpy as np
import time

start_time = time.time()

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

learning_rate = .01
iter_1 = 50
iter_2 = 50
counter = 0


class NeuralNetwork:

    def __init__(self, rows, columns=0):
        self.mtrx = np.zeros((rows, 1))
        self.weight = np.random.normal(size=(rows, columns)) / (columns ** .5)
        self.bias = np.random.random((rows, 1))
        self.grad = np.zeros((rows, columns))

    def sigmoid(self):
        return 1 / (1 + np.exp(-self.mtrx))

    def sigmoid_derivative(self):
        return self.sigmoid() * (1.0 - self.sigmoid())

lvl_input = NeuralNetwork(784)
lvl_one = NeuralNetwork(200, 784)
lvl_two = NeuralNetwork(200, 200)
lvl_output = NeuralNetwork(10, 200)


def forward_prop():
    lvl_one.mtrx = lvl_one.weight.dot(lvl_input.mtrx)
    lvl_two.mtrx = lvl_two.weight.dot(lvl_one.sigmoid())
    lvl_output.mtrx = lvl_output.weight.dot(lvl_two.sigmoid())


def back_prop(actual):
    val = np.zeros((10, 1))
    val[actual] = 1

    error_1 = lvl_output.sigmoid() - val
    lvl_output.grad += error_1 @ lvl_one.sigmoid().T

    error_2 = lvl_two.sigmoid_derivative() * (lvl_output.weight.T @ error_1)
    lvl_two.grad += error_2 @ lvl_one.sigmoid().T

    error_3 = lvl_one.sigmoid_derivative() * (lvl_two.weight.T @ error_2)
    lvl_one.grad += error_3 @ lvl_input.mtrx.T


def make_image(c):
    lvl_input.mtrx = training_images[c]


def cost(actual):
    val = np.zeros((10, 1))
    val[actual] = 1
    cost_val = (lvl_output.sigmoid() - val) ** 2
    return np.sum(cost_val)


def update():
    lvl_output.weight -= learning_rate * lvl_output.grad
    lvl_two.weight -= learning_rate * lvl_two.grad
    lvl_one.weight -= learning_rate * lvl_one.grad
    lvl_output.grad = np.zeros(np.shape(lvl_output.grad))
    lvl_two.grad = np.zeros(np.shape(lvl_two.grad))
    lvl_one.grad = np.zeros(np.shape(lvl_one.grad))


for batch_num in range(iter_1):
    update()
    for batches in range(iter_2):
        make_image(counter)
        num = np.argmax(training_labels[counter])
        counter += 1
        forward_prop()
        back_prop(num)
        print("actual: ", num, "     guess: ", np.argmax(lvl_output.mtrx), "     cost", cost(num))

print("--- %s seconds ---" % (time.time() - start_time))
