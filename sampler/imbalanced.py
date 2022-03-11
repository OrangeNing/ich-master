import torch
import numpy as np

# import random
# import numpy as np
#
# from torch.utils.data.sampler import Sampler
#
#
# class MultilabelBalancedRandomSampler(Sampler):
#     """
#     MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
#     number of classes n_classes, samples from the data with equal probability per class
#     effectively oversampling minority classes and undersampling majority classes at the
#     same time. Note that using this sampler does not guarantee that the distribution of
#     classes in the output samples will be uniform, since the dataset is multilabel and
#     sampling is based on a single class. This does however guarantee that all classes
#     will have at least batch_size / n_classes samples as batch_size approaches infinity
#     """
#
#     def __init__(self, labels, indices=None, class_choice="least_sampled"):
#         """
#         Parameters:
#         -----------
#             labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
#             indices: an arbitrary-length 1-dimensional numpy array representing a list
#             of indices to sample only from
#             class_choice: a string indicating how class will be selected for every
#             sample:
#                 "least_sampled": class with the least number of sampled labels so far
#                 "random": class is chosen uniformly at random
#                 "cycle": the sampler cycles through the classes sequentially
#         """
#         self.labels = labels
#         self.indices = indices
#         if self.indices is None:
#             self.indices = range(len(labels))
#
#         self.num_classes = self.labels.shape[1]
#
#         # List of lists of example indices per class
#         self.class_indices = []
#         for class_ in range(self.num_classes):
#             lst = np.where(self.labels[:, class_] == 1)[0]
#             lst = lst[np.isin(lst, self.indices)]
#             self.class_indices.append(lst)
#
#         self.counts = [0] * self.num_classes
#
#         assert class_choice in ["least_sampled", "random", "cycle"]
#         self.class_choice = class_choice
#         self.current_class = 0
#
#     def __iter__(self):
#         self.count = 0
#         return self
#
#     def __next__(self):
#         if self.count >= len(self.indices):
#             raise StopIteration
#         self.count += 1
#         return self.sample()
#
#     def sample(self):
#         class_ = self.get_class()
#         class_indices = self.class_indices[class_]
#         chosen_index = np.random.choice(class_indices)
#         if self.class_choice == "least_sampled":
#             for class_, indicator in enumerate(self.labels[chosen_index]):
#                 if indicator == 1:
#                     self.counts[class_] += 1
#         return chosen_index
#
#     def get_class(self):
#         if self.class_choice == "random":
#             class_ = random.randint(0, self.labels.shape[1] - 1)
#         elif self.class_choice == "cycle":
#             class_ = self.current_class
#             self.current_class = (self.current_class + 1) % self.labels.shape[1]
#         elif self.class_choice == "least_sampled":
#             min_count = self.counts[0]
#             min_classes = [0]
#             for class_ in range(1, self.num_classes):
#                 if self.counts[class_] < min_count:
#                     min_count = self.counts[class_]
#                     min_classes = [class_]
#                 if self.counts[class_] == min_count:
#                     min_classes.append(class_)
#             class_ = np.random.choice(min_classes)
#         return class_
#
#     def __len__(self):
#         return len(self.indices)
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import random
import numpy as np

from torch.utils.data.sampler import Sampler


class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, labels, indices=None, class_choice="least_sampled"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from
            class_choice: a string indicating how class will be selected for every
            sample:
                "least_sampled": class with the least number of sampled labels so far
                "random": class is chosen uniformly at random
                "cycle": the sampler cycles through the classes sequentially
        """
        self.labels = labels
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))

        self.num_classes = self.labels.shape[1]

        # List of lists of example indices per class
        self.class_indices = []
        for class_ in range(self.num_classes):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

        self.counts = [0] * self.num_classes

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = np.random.choice(class_indices)
        if self.class_choice == "least_sampled":
            for class_, indicator in enumerate(self.labels[chosen_index]):
                if indicator == 1:
                    self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class + 1) % self.labels.shape[1]
        elif self.class_choice == "least_sampled":
            min_count = self.counts[0]
            min_classes = [0]
            for class_ in range(1, self.num_classes):
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = np.random.choice(min_classes)
        return class_

    def __len__(self):
        return len(self.indices)

class RandomDataset(Dataset):
    def __init__(self, n_examples, n_features, n_classes, mean_labels_per_example):
        self.n_examples = n_examples
        self.n_features = n_features
        self.n_classes = n_classes
        self.X = np.random.random([self.n_examples, self.n_features])

        class_probabilities = np.random.random([self.n_classes])
        class_probabilities = class_probabilities / sum(class_probabilities)
        class_probabilities *= mean_labels_per_example
        self.y = (
            np.random.random([self.n_examples, self.n_classes]) < class_probabilities
        ).astype(int)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, index):
        example = Variable(torch.tensor(self.X[index]), requires_grad=False)
        labels = Variable(torch.tensor(self.y[index]), requires_grad=False)
        return {"example": example, "labels": labels}


def get_data_loaders(batch_size, val_size):
    dataset = RandomDataset(20000, 100, 20, 2)

    # Split into training and validation
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(val_size * len(dataset)))
    train_idx, validate_idx = indices[split:], indices[:split]

    train_sampler = MultilabelBalancedRandomSampler(
        dataset.y, train_idx, class_choice="least_sampled"
    )
    validate_sampler = SubsetRandomSampler(validate_idx)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,)
    validate_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=validate_sampler,
    )
    return train_loader, validate_loader


def main():
    epochs = 2
    train_loader, validate_loader = get_data_loaders(batch_size=512, val_size=0.2)

    for epoch in range(epochs):
        print("================ Training phase ===============")
        for batch in train_loader:
            examples = batch["example"]
            labels = batch["labels"]
            print("Label counts per class:")
            sum_ = labels.sum(axis=0)
            print(sum_)
            print("Difference between min and max")
            print(max(sum_) - min(sum_))
            print("")
        print("")

        print("=============== Validation phase ==============")
        for batch in validate_loader:
            examples = batch["example"]
            labels = batch["labels"]
            print("Label counts per class:")
            sum_ = labels.sum(axis=0)
            print(sum_)
            print("Difference between min and max")
            print(max(sum_) - min(sum_))
            print("")
        print("")


if __name__ == "__main__":
    main()