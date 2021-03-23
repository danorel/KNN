import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider


class KNNDashboard:
    def __init__(self, classifier, x, y, title='Training set'):
        self.__classifier = classifier
        self.__title = title
        self.__l = None
        self.__fig = None
        self.__slider_k = None
        self.__slider_size = None
        self.__x = x
        self.__y = y

    def render_map(self) -> None:
        x_1, x_2 = np.meshgrid(
            np.arange(start=self.__x[:, 0].min() - 1,
                      stop=self.__x[:, 0].max() + 1,
                      step=0.01),
            np.arange(start=self.__x[:, 1].min() - 1,
                      stop=self.__x[:, 1].max() + 1,
                      step=0.01))

        plt.contourf(x_1, x_2, self.__classifier.predict(
            np.array([x_1.ravel(), x_2.ravel()]).T).reshape(x_1.shape),
            alpha=0.75,
            cmap=ListedColormap(('red', 'green', 'black', 'blue')))

        plt.xlim(x_1.min(), x_1.max())
        plt.ylim(x_2.min(), x_2.max())

        for i, j in enumerate(np.unique(self.__y)):
            plt.scatter(self.__x[self.__y == j, 0], self.__x[self.__y == j, 1],
                        c=ListedColormap(('red', 'green', 'black', 'blue'))(i),
                        label=j)

        plt.title(self.__title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

    def render_graph(self) -> None:

        """
        Inner method for update tracking
        :return: None
        """
        def update(val) -> None:
            k = self.__slider_k.val
            size = self.__slider_size.val
            self.__l.set_ydata(self.__y)
            # Re-drawing the figure
            self.__fig.canvas.draw()
            return None

        # Plotting
        self.__fig = plt.figure()

        plt.subplots_adjust(bottom=0.25)
        ax = self.__fig.subplots()

        # Create and plot current xs, ys
        l, = ax.plot(self.__x[0], self.__x[1], label=self.__title)
        self.__l = l

        ax_k    = plt.axes([0.25, 0.15, 0.65, 0.03])
        ax_size = plt.axes([0.25, 0.1, 0.65, 0.03])

        self.__slider_k    = Slider(ax_k, 'K', 1, 50, valinit=1)
        self.__slider_size = Slider(ax_size, 'Size', 1, 2500, valinit=1000)

        self.__slider_k.on_changed(update)
        self.__slider_size.on_changed(update)

        plt.grid()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

        return None
