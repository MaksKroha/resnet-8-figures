import torchvision
import torch.utils.data as tdata


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST("data/mnist/", True, transforms, download=True)
test_dataset = torchvision.datasets.MNIST("data/mnist/", False, transforms, download=True)


class Loader:
    def __init__(self, batch_size):
        self.train_dataloader = tdata.DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_dataloader = tdata.DataLoader(test_dataset, 1)

    def getTrainLoader(self):
        return self.train_dataloader

    def getTestLoader(self):
        return self.test_dataloader
