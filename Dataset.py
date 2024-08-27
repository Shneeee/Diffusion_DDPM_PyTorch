import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size, shuffle=True):
    dataset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader