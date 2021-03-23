import abc


class AbstractRandomValueGenerator(abc.ABC):
    @abc.abstractmethod
    def rand(self):
        """
        Method for extracting the negative random value with settled minimum and maximum values.
        """
        pass


class AbstractRandomListGenerator(abc.ABC):
    @abc.abstractmethod
    def rand(self, size: int):
        """
        Method for generating negative stream of values.
        """
        pass


class AbstractRandomDatasetGenerator(abc.ABC):
    @abc.abstractmethod
    def commit(self, size: int):
        """
        Method for generating random points without defining classes.
        Generates 'size' points in format: [(x, y)]
        :return [(float, float)]
        """
        pass

    @abc.abstractmethod
    def commit_classified(self, size: int, classes: int):
        """
        Method for generating random points with class definitions.
        Generates 'size' points in format: ((x, y), class)
        :return [((float, float), int)]
        """
        pass
