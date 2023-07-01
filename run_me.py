import argparse

# dataset
from pandas import read_csv
from torch.utils.data import Dataset
from torch.utils.data import random_split

# dataloader
from torch.utils.data import DataLoader

# model
from torch.nn import Module
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, required=True, help="the name of the training datafile")
    args = vars(parser.parse_args())
    return args

# dataset definition
class titanic_dataset(Dataset):
    # load the dataset
    def __init__(self, path):
        df = read_csv(path, sep=",", header=0)
        # store the inputs and outputs
        self.data   = df.values[:, 2:]
        self.labels = df.values[:, 1]
        #print(self.data)
        #print(self.labels)

    # number of rows in the dataset
    def __len__(self):
        return len(self.data)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.data[idx], self.labels[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.12):
        test_size = round(n_test * len(self.data))
        train_size = len(self.data) - test_size
        return random_split(self, [train_size, test_size])

# define model
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X

# prepare the dataset
def prepare_data(path):
    dataset = titanic_dataset(path)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

def main():
    args = get_args()
    train_dl, test_dl = prepare_data(args["data"])
    #print(len(train_dl.dataset), len(test_dl.dataset))
    model = MLP(10)

if __name__ == "__main__":
    main()
