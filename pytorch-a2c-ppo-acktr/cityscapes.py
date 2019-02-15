import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import os.path
import glob


class Cityscapes(data.Dataset):
    """Cityscapes Dataset
    """

    def __init__(self, root, train, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = []
        self.masks = []
        if train:
            folders = ['train']
        else:
            folders = ['test']
        for folder in folders:
            self.path = os.path.join(self.root, folder)
            for filename in glob.iglob(self.path + '/**/*color.png', recursive=True):
                self.imgs.append(filename)
                mask_file = filename.strip('color.png') + 'labelIds.png'
                self.masks.append(mask_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img = Image.open(self.imgs[index]).convert('RGB').resize((256, 256))
        if self.transform is not None:
            img = self.transform(img)

        target = Image.open(self.masks[index])
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = np.array(target.getdata()).reshape(
            target.size[0], target.size[1])

        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
