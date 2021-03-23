import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class KNNDashboard:
    def __init__(self, classifier, x, y):
        self.__x = x
        self.__y = y
        self.__classifier = classifier

    def render(self):
        x_1, x_2 = np.meshgrid(
            np.arange(start=self.__x[:, 0].min() - 1,
                      stop=self.__y[:, 0].max() + 1,
                      step=0.01),
            np.arange(start=self.__x[:, 1].min() - 1,
                      stop=self.__x[:, 1].max() + 1,
                      step=0.01))

        plt.contourf(x_1, x_2, self.__classifier.predict(
            np.array([x_1.ravel(), x_2.ravel()]).T).reshape(x_1.shape),
            alpha=0.75,
            cmap=ListedColormap(('red', 'green')))

        plt.xlim(x_1.min(), x_1.max())
        plt.ylim(x_2.min(), x_2.max())

        for i, j in enumerate(np.unique(self.__y)):
            plt.scatter(self.__x[self.__y == j, 0], self.__x[self.__y == j, 1],
                        c=ListedColormap(('red', 'green'))(i),
                        label=j)

        plt.title('Classifier (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()
