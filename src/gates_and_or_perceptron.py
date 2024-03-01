from activation import unit_step
from perceptron_monolayer import Perceptron


# Logic Gate AND
perceptron = Perceptron(inputs=2, activation_function=unit_step)
perceptron.set_weights([2, 2])
perceptron.set_bias(-3)

print(f"0 0 = {perceptron.run([0, 0]):.5f}")
print(f"0 1 = {perceptron.run([0, 1]):.5f}")
print(f"1 0 = {perceptron.run([1, 0]):.5f}")
print(f"1 1 = {perceptron.run([1, 1]):.5f}")

# Logic Gate OR
perceptron = Perceptron(inputs=2, activation_function=unit_step)
perceptron.set_weights([3, 3])
perceptron.set_bias(-1)

print(f"0 0 = {perceptron.run([0, 0]):.5f}")
print(f"0 1 = {perceptron.run([0, 1]):.5f}")
print(f"1 0 = {perceptron.run([1, 0]):.5f}")
print(f"1 1 = {perceptron.run([1, 1]):.5f}")
