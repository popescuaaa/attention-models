import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

stocks = ['GOOG', 'APPL', 'MSFT', 'AMZN']
stocks_map = {
    'GOOG': 1,
    'APPL': 2,
    'MSFT': 3,
    'AMZN': 4
}

class Stocks(Dataset):
    def __init__(self):
        super(Stocks, self).__init__()
        self.raw_data = pd.read_csv('./data.csv').values
        self.data = []
        for stock in stocks:
            for year in range(2000, 2011):
                mask = [False] * len(self.raw_data)
                for idx in range(len(self.raw_data)):
                    if self.raw_data[idx][0] == stock and int(self.raw_data[idx][1].split(' ')[-1]) == year:
                        mask[idx] = True

                _data = self.raw_data[mask]
                self.data.append(
                    torch.from_numpy(
                        np.array(list(map(lambda de: (stocks_map[de[0]], de[2]), _data)))
                    )
                )

        self.data = list(filter(lambda d: len(d) == 12, self.data))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    ds = Stocks()
    dl = DataLoader(ds, batch_size=5, shuffle=True, num_workers=2)
    for i, e in enumerate(dl):
        print(type(e))
