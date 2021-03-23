import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class KNNDashboard:
    def __init__(self, classifier, title='Training set'):
        self.__classifier = classifier
        self.__title = title

    def render(self, x, y):
        x_1, x_2 = np.meshgrid(
            np.arange(start=x[:, 0].min() - 1,
                      stop=x[:, 0].max() + 1,
                      step=0.01),
            np.arange(start=x[:, 1].min() - 1,
                      stop=x[:, 1].max() + 1,
                      step=0.01))

        plt.contourf(x_1, x_2, self.__classifier.predict(
            np.array([x_1.ravel(), x_2.ravel()]).T).reshape(x_1.shape),
            alpha=0.75,
            cmap=ListedColormap(('red', 'green', 'black', 'blue')))

        plt.xlim(x_1.min(), x_1.max())
        plt.ylim(x_2.min(), x_2.max())

        for i, j in enumerate(np.unique(y)):
            plt.scatter(x[y == j, 0], x[y == j, 1],
                        c=ListedColormap(('red', 'green', 'black', 'blue'))(i),
                        label=j)

        plt.title(self.__title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
