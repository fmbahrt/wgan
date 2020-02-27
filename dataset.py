import torch
import torchvision
import torchvision.transforms as transforms

def celeba(path, batch_size=64):
    transformation = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(path, transformation)
    sampler = torch.utils.data.SubsetRandomSampler(range(len(dataset)))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=8,
        drop_last=True
    )

    return dataloader

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    it = iter(celeba('/home/frederik/Documents/diku/bscthesis/data/celeba'))
    batch = next(it)

    img = np.transpose(vutils.make_grid(batch[0], padding=2,
                                             normalize=True), (1, 2, 0))
    
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("celebA")
    plt.imshow(img)
    plt.show()
    print("bæ")
