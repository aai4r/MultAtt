import torch
import torchvision
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
import numpy as np
import matplotlib.pyplot as plt
from data.dataset import SimpleDataset
from abc import abstractmethod

class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type == 'CenterCrop':
            return method(self.image_size) 
        elif transform_type == 'Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        elif transform_type == 'Resize':
            return method((self.image_size, self.image_size))
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            # transform_list = ['Resize', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
            transform_list = ['Resize', 'RandomHorizontalFlip', 'ToTensor']
        else:
            # transform_list = ['Resize', 'ToTensor', 'Normalize']
            transform_list = ['Resize', 'ToTensor']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_path, load_set, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, num_workers):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_path, load_set, aug):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_path, load_set, transform)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

if __name__ == '__main__':
    datamgr = SimpleDataManager(image_size=224, batch_size=1, num_workers=0)
    dataloader = datamgr.get_data_loader(data_path='/work/CAER-S', load_set='test', aug=False)
    for i, (x1, x2, y) in enumerate(dataloader):
        if i % 2 == 0:
            img = torchvision.utils.make_grid(x1).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()

            img = torchvision.utils.make_grid(x2).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
