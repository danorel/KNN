from src.dataset import RandomPointGenerator

if __name__ == '__main__':
    from src.kit import knn
    training_dataset = RandomPointGenerator(1000).classify()
    print(training_dataset[0:50])
    knn(1)
