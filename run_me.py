import argparse
from pandas import read_csv
from torch.utils.data import Dataset

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

# prepare the dataset
def prepare_data(path):
    dataset = titanic_dataset(args["data"])


def main():
    args = get_args()
    dataset = titanic_dataset(args["data"])
    
if __name__ == "__main__":
    main()
