import abc


class AbstractDisplayKNN(abc.ABC):
    @abc.abstractmethod
    def render_map(self):
        pass

    @abc.abstractmethod
    def render_graph(self):
        pass
