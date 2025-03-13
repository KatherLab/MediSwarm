import numpy as np
import uuid
import torch
import torch.utils.data as data

class MiniDatasetForTesting(data.Dataset):
    def __init__(self):
        num_entries = 10
        self.data = [{'uid': str(uuid.uuid4()), 'source': self.dummy_image(index), 'target': index % 2} for index in range(num_entries)]

    @staticmethod
    def dummy_image(index):
        shape = (1, 18, 18)
        dtype = np.float16
        if index % 2 == 0:
            array = np.zeros(shape, dtype=dtype)
            array[0, 0, index] = 1
        else:
            array = np.ones(shape, dtype=dtype)
            array[0, 0, index] = 0
        return torch.from_numpy(array)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return[i['target'] for i in self.data]
