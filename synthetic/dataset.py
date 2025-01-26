from torch.utils.data import Dataset


class PlaceholderDataset(Dataset):
    def __init__(
        self,
        num: int,
    ):
        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return idx
