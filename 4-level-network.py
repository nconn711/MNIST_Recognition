import numpy as np
import time

start_time = time.time()

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']


class NeuralNetwork:

    def __init__(self, rows, columns=0):
        self.mtrx = np.zeros((rows, 1))
        self.weight = np.random.normal(size=(rows, columns)) / (columns ** .5)
        self.bias = np.random.normal(size=(rows, 1))
        self.grad = np.zeros((rows, columns))

    def sigmoid(self):
        return 1 / (1 + np.exp(-self.mtrx))

    def sigmoid_derivative(self):
        return self.sigmoid() * (1.0 - self.sigmoid())

lvl_input = NeuralNetwork(784)
lvl_one = NeuralNetwork(20, 784)
lvl_two = NeuralNetwork(20, 20)
lvl_output = NeuralNetwork(10, 20)


def forward_prop():
    lvl_one.mtrx = lvl_one.weight.dot(lvl_input.mtrx) + lvl_one.bias
    lvl_two.mtrx = lvl_two.weight.dot(lvl_one.sigmoid()) + lvl_two.bias
    lvl_output.mtrx = lvl_output.weight.dot(lvl_two.sigmoid()) + lvl_output.bias


def back_prop(actual):
    val = np.zeros((10, 1))
    val[actual] = 1

    delta_3 = (lvl_output.sigmoid() - val) * lvl_output.sigmoid_derivative()
    delta_2 = np.dot(lvl_output.weight.transpose(), delta_3) * lvl_two.sigmoid_derivative()
    delta_1 = np.dot(lvl_two.weight.transpose(), delta_2) * lvl_one.sigmoid_derivative()

    lvl_output.grad = lvl_two.sigmoid().transpose() * delta_3
    lvl_two.grad = lvl_one.sigmoid().transpose() * delta_2
    lvl_one.grad = lvl_input.mtrx.transpose() * delta_1

    lvl_output.bias -= learning_rate * delta_3
    lvl_two.bias -= learning_rate * delta_2
    lvl_one.bias -= learning_rate * delta_1


def make_image_train(c):
    lvl_input.mtrx = training_images[c]


def make_image_test(c):
    lvl_input.mtrx = test_images[c]


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


learning_rate = .2
iter_1 = 50000
iter_2 = 1
counter = 0
print("Learning . . . . \n")
for batch_num in range(iter_1):
    update()
    for batches in range(iter_2):
        make_image_train(counter)
        num = np.argmax(training_labels[counter])
        counter += 1
        forward_prop()
        back_prop(num)
print("Done learning.\n")

print("Testing . . . .\n")
iter_1 = 10000
correct = 0
counter = 0
for test_cases in range(iter_1):
    make_image_test(counter)
    num = np.argmax(test_labels[counter])
    counter += 1
    forward_prop()
    if np.argmax(lvl_output.mtrx) == num:
        correct += 1
    #print("Actual: {}".format(num), "    Guess: {}".format(np.argmax(lvl_output.mtrx)), "    Cost: {}".format(cost(num)))
print("\nPercent Correct: {}%".format((correct * 100)/iter_1))
print("\n--- %s seconds ---" % (time.time() - start_time))
