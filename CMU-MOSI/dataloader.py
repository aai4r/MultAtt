import glob
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import csv
import pdb

def multi_collate(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[1].shape[0], reverse=True)

    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    language = torch.cat([sample[0].unsqueeze(0) for sample in batch], dim=0)
    acoustic = pad_sequence([sample[1] for sample in batch])
    visual = pad_sequence([sample[2] for sample in batch])
    label = torch.cat([torch.tensor(sample[3]).unsqueeze(0) for sample in batch], dim=0)

    return language, acoustic, visual, label

class MOSI_Dataset(Dataset):
    def __init__(self, data_path, d_type, finetuned=False, sequential=True):
        self.data_path = data_path

        if finetuned:
            self.language_list = glob.glob(os.path.join(self.data_path, d_type, 'language_finetuned', '*.pt'))
        else:
            self.language_list = glob.glob(os.path.join(self.data_path, d_type, 'language', '*.pt'))

        file = os.path.join(self.data_path, d_type + '.tsv')
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            self.label_dict = {}
            for (i, line) in enumerate(reader):
                if i == 0:
                    continue
                self.label_dict.update({line[0]: line[1]})

        self.sequential = sequential

    def __getitem__(self, index):
        language = self.language_list[index]
        id = language.split('/')[-1].split('_')[-1][:-3]

        acoustic = language.split('/')
        acoustic[-2] = 'acoustic'
        acoustic[-1] = 'acoustic_' + str(id) + '.pt'
        acoustic = '/'.join(acoustic)

        visual = language.split('/')
        visual[-2] = 'visual'
        visual[-1] = 'visual_' + str(id) + '.pt'
        visual = '/'.join(visual)

        language, acoustic, visual = torch.load(language, map_location=torch.device('cpu')), \
                                     torch.load(acoustic, map_location=torch.device('cpu')), \
                                     torch.load(visual, map_location=torch.device('cpu'))
        if not self.sequential:
            acoustic, visual = torch.mean(acoustic, 0), torch.mean(visual, 0)

        label = float(self.label_dict[str(id)])
        return language, acoustic, visual, label

    def __len__(self):
        return len(self.language_list)


if __name__ == "__main__":
    train_data = MOSI_Dataset('/work/MOSI', 'train', finetuned=False, sequential=False)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=False, num_workers=4, collate_fn=multi_collate)

    for i, batch in enumerate(train_loader):
        l, a, v, y = batch
        print(l.shape)
        print(a.shape)
        print(v.shape)
        print(y)
        pdb.set_trace()
