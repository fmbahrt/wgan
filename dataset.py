import os
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

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

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
    '.ppm', '.PPM', '.bmp', '.BMP'
]

NP_EXTENSIONS = [
    '.npy'
]

def is_valid_file(path, mode='img'):
    if mode == 'img':
        return any([path.endswith(ext) for ext in IMG_EXTENSIONS])
    elif mode == 'np':
        return any([path.endswith(ext) for ext in NP_EXTENSIONS])
    else:
        return False # What please

def get_fps(dirs, mode='img'):
    if not isinstance(dirs, list):
        dirs = [dirs]

    img_paths = []
    for d in dirs:
        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_valid_file(fname, mode=mode):
                    path = os.path.join(root, fname)
                    img_paths.append(path)
    return img_paths

class TwoAFCDataset(data.Dataset):

    def __init__(self, dataroot, loadsize=64, tf=True):
        if not isinstance(dataroot, list):
            dataroot = [dataroot]
        self.dataroot = dataroot

        # Load Dataset - paths into memory ;-)
        self.tf = tf
        # Load Reference Images
        self.dir_ref = [os.path.join(root, 'ref') for root in self.dataroot]
        self.ref_paths = get_fps(self.dir_ref, mode='img')
        self.ref_paths = sorted(self.ref_paths)

        # Load p0
        self.dir_p0 = [os.path.join(root, 'p0') for root in self.dataroot]
        self.p0_paths = get_fps(self.dir_p0, mode='img')
        self.p0_paths = sorted(self.p0_paths)

        # Load p1
        self.dir_p1 = [os.path.join(root, 'p1') for root in self.dataroot]
        self.p1_paths = get_fps(self.dir_p1, mode='img')
        self.p1_paths = sorted(self.p1_paths)

        # Load judgements
        self.dir_judge = [os.path.join(root, 'judge') for root in self.dataroot]
        self.judge_paths = get_fps(self.dir_judge, mode='np')
        self.judge_paths = sorted(self.judge_paths)

        # Transformations
        transform_list = [
            transforms.Resize(loadsize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.p0_paths)

    def __getitem__(self, idx):
        p0_path = self.p0_paths[idx]
        p0_img  = Image.open(p0_path).convert('RGB')

        p1_path = self.p1_paths[idx]
        p1_img  = Image.open(p1_path).convert('RGB')

        ref_path = self.ref_paths[idx]
        ref_img  = Image.open(ref_path).convert('RGB')

        judge_path = self.judge_paths[idx]
        #judge_lbl  = np.load(judge_path).reshape((1, 1, 1, ))
        judge_lbl  = np.load(judge_path)
         
        if self.tf:
            p0_img = self.transform(p0_img)
            p1_img = self.transform(p1_img)
            ref_img = self.transform(ref_img)
            judge_lbl = torch.FloatTensor(judge_lbl)

        return {
            'ref'   : ref_img,
            'p0'    : p0_img,
            'p1'    : p1_img,
            'judge' : judge_lbl
        }

def twoafc(dataroots, batch_size=8):
    dataset    = TwoAFCDataset(dataroots)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    return dataloader

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    it = iter(twoafc('./dataset/2afc/train/traditional'))
    batch = next(it)

    ref = batch['ref'][0].numpy().transpose(2,1,0)
    p0  = batch['p0'][0].numpy().transpose(2,1,0)
    p1  = batch['p1'][0].numpy().transpose(2,1,0)
    print(batch['judge'][0]) 
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(p0)
    ax[1].imshow(ref)
    ax[2].imshow(p1)
    plt.show()

    #dataset = TwoAFCDataset('./dataset/2afc/train/traditional', tf=False)
    #
    #for i in range(20):
    #    item = dataset.__getitem__(i)
    #    print(item['judge'])
    #    f, ax = plt.subplots(1, 3)
    #    ax[0].imshow(item['p0'])
    #    ax[1].imshow(item['ref'])
    #    ax[2].imshow(item['p1'])
    #    plt.show()

    #it = iter(celeba('/home/frederik/Documents/diku/bscthesis/data/celeba'))
    #batch = next(it)

    #img = np.transpose(vutils.make_grid(batch[0], padding=2,
    #                                         normalize=True), (1, 2, 0))
    #
    #plt.figure(figsize=(8,8))
    #plt.axis("off")
    #plt.title("celebA")
    #plt.imshow(img)
    #plt.show()
    #print("bæ")
