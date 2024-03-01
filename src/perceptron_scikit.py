import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron


# Digits dataset
digits = load_digits()
print(digits.data.shape)

figure = plt.figure()
for i in range(12):
    ax = figure.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

plt.show()

# Perceptron training
X, y = load_digits(return_X_y=True)
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
print(clf.score(X, y))
