from torch.utils.data import Dataset

def wrap_tensor_only(dataset):
    """
    Wraps any dataset so that it returns only the first element (typically the image tensor).
    
    Args:
        dataset (Dataset): A PyTorch dataset that returns a tuple (image, label, ...).

    Returns:
        Dataset: A dataset that returns only the image tensor.
    """
    class TensorOnlyWrapper(Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            item = self.base_dataset[idx]
            return item[0] if isinstance(item, (tuple, list)) else item

    return TensorOnlyWrapper(dataset)