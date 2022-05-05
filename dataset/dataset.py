import oneflow as flow
from flowvision import transforms
from flowvision import datasets
import oneflow as flow
from oneflow.utils.data import Dataset
from libai.data.structures import DistTensorData, Instance


class MnistDataSet(Dataset):
    def __init__(self, path, is_train, indc=None):
        print(path)
        self.data = datasets.MNIST(
            root=path,
            train=is_train,
            transform=transforms.ToTensor(),
            download=False
        )
        if indc is not None:
            self.data = flow.utils.data.Subset(dataset=self.data, indices=range(indc))

    def __getitem__(self, idx):
        sample = Instance(
            inputs=DistTensorData(
                flow.tensor(self.data[idx][0], dtype=flow.float32)
            ),
            labels=DistTensorData(
                flow.tensor(self.data[idx][1], dtype=flow.int), placement_idx=-1)
        )

        return sample

    def __len__(self):
        return len(self.data)
