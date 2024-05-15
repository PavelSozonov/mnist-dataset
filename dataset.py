from torchvision import datasets, transforms

datasets.MNIST("dataset", download=True, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
datasets.MNIST("dataset", download=True, train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
