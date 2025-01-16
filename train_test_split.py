from sklearn.model_selection import train_test_split


def split_data(data):
    train, test = train_test_split(data, test_size=0.3)

    train_train_datasets = []
    train_val_datasets = []
    for _ in range(5):
        train_train, train_val = train_test_split(train, test_size=0.2)
        train_train_datasets.append(train_train)
        train_val_datasets.append(train_val)

    return train_train_datasets, train_val_datasets, test