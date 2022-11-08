import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class Logistic_Regression:
    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def __init__(self, X, Y):
        self.X = np.insert(X, 0, [1] * X.shape[0], axis=1)
        self.Y = Y
        self.b = np.zeros((self.X[0].shape[0]))

    def update_coeffs(self, learning_rate):
        Y_pred = self.predict()
        Y = self.Y
        p = len(Y)

        sum = np.zeros((len(self.b)))
        for i, x in enumerate(self.X):
            temp = 0
            for j, k in enumerate(self.X[i]):
                temp += self.b[j] * k
            for coef in range(len(x)):
                sum[coef] += -(1 / p) * x[coef] * (Y[i] - self.sigmoid(temp))
        self.b -= learning_rate * sum

    # self.b[coef] - (learning_rate *
    def predict(self, X=None):
        Y_pred = []
        if X is None:
            X = self.X
        else:
            X = np.insert(X, 0, [1] * X.shape[0], axis=1)
        b = self.b
        for i in range(X.shape[0]):
            pred_value = np.dot(np.array(b).T, np.array(X[i]))
            Y_pred.append(pred_value)

        return Y_pred

    def compute_cost(self, Y_pred):
        p = len(self.Y)
        cross_entropy_loss = 0
        for i in range(p):
            cross_entropy_loss += -self.Y[i] * np.log(self.sigmoid(Y_pred[i])) - (
                    1 - self.Y[i]) * np.log(1 - self.sigmoid(Y_pred[i]))
        return cross_entropy_loss / p


iris = datasets.load_iris(as_frame=True)
# get labels (for 2 classes only) and features
y = iris.target
x = iris.data
y = y[y < 2]
x = x[['petal width (cm)', 'petal length (cm)']]
x = x.iloc[y.index].values
# print(x)
y = y.values.astype('float')
# print(y)

regressor = Logistic_Regression(x, y)
iterations = 0
steps = 20000
learning_rate = 0.1
costs = []

while iterations < steps:
    Y_pred = regressor.predict()
    cost = regressor.compute_cost(Y_pred)
    costs.append(cost)
    regressor.update_coeffs(learning_rate)
    # print(regressor.b)
    iterations += 1
print(regressor.b)
#
# # gradient descent function
# def gradient_descent(g, step, max_its, w, p):
#     # compute gradient
#     gradient = grad(g)
#     # gradient descent loop
#     weight_history = [w]  # weight history container
#     cost_history = [g(w)]  # cost history container
#     for k in range(max_its):
#         # eval gradient
#         grad_eval = gradient(w)
#         grad_eval_norm = grad_eval / np.linalg.norm(grad_eval)
#         # take grad descent step
#         if step == 'd':  # diminishing step
#             alpha = 1 / (k + 1)
#         else:  # constant step
#             alpha = step
#         w = w - alpha * grad_eval_norm
#         # record weight and cost
#         weight_history.append(w)
#         cost_history.append(g(w))
#     return weight_history, cost_history
#
# def gradient():
#
#
# import numpy as np
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# from sklearn import datasets
#
#
# # define sigmoid function
# def sigmoid(t):
#     return 1 / (1 + np.exp(-t))
#
#
# # compute linear combination of input point
# def model(x, w):
#     print("first", f'{w}')
#     a = (w[0] + np.dot(x, w[1:]))
#     return a.T
#
#
# # # the convex cross-entropy cost function
#
# def cross_entropy(w, x, y):
#     # compute sigmoid of model
#     a = sigmoid(model(x, w))
#     # compute cost of label 0 points
#     #     a=a.reshape((a.shape[1],1))
#     ind = np.argwhere(y == 0)[:, 0]
#     #     print("ind",ind)
#     #     print("ind size", a[:,ind])
#     #     print (a[:,ind])
#     cost = -np.sum(np.log(1 - a[:, ind]))
#     print(cost)
#     # add cost of label 0 points
#     ind = np.argwhere(y == 1)[:, 0]
#     cost -= np.sum(np.log(a[:, ind]))
#     print(cost)
#     #     a=a.reshape((1,a.shape[0]))
#     # compute cross-entropy
#     return cost / y.size
#
#
# iris = datasets.load_iris(as_frame=True)
# # get labels (for 2 classes only) and features
# y = iris.target
# x = iris.data
# y = y[y < 2]
# x = x[['petal width (cm)', 'petal length (cm)']]
# x = x.iloc[y.index].values
# y = y.values.astype('float')
# # print(y)
#
# scatter plot
colors = ('r', 'b')
for target in range(2):
    plt.scatter(x[y == target, 0], x[y == target, 1],
                c=colors[target])
    plt.grid(1)
    plt.legend([iris.target_names[0], iris.target_names[1]])
    plt.xlabel(iris.feature_names[2])
    plt.ylabel(iris.feature_names[3])

# plot boundary
x1p = np.linspace(-50, 50, 10)
print(x1p)
x2p = -(regressor.b[0] + regressor.b[1] * x1p) / regressor.b[2]
# x2p = np.linspace(1, 2, 5)
print(x2p)
plt.plot(x1p, x2p, c = 'g')
plt.xlim(1, 5)
plt.ylim(0, 2)
plt.show()
#
#
# def c(t):
#     c = cross_entropy(t, x, y)
#     return c
#
#
# iter = 100
# w = np.array([[1.], [1.], [1.]])
# a, b = gradient_descent(c, 0.1, iter, w, 0)
# plt.figure(0)
# plt.plot(b)
#
# plt.figure(1)
# xp = np.array([np.linspace(0, 3, 20)])
# xp = xp.reshape(-1, 1)
# plt.plot(xp, sigmoid(model(xp, a[iter])))
