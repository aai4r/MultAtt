from PIL import Image
import os
import itertools

class SimpleDataset:
    def __init__(self, data_path, load_set, transform):
        self.data_path = data_path
        self.load_set = load_set
        self.data_file = os.path.join(self.data_path, self.load_set)
        self.emo_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.file = []
        for i in range(len(self.emo_list)):
            self.file.append([i_id.strip() for i_id in open(os.path.join(self.data_file, self.emo_list[i] + '.txt'))])
        self.file = list(itertools.chain(*self.file))
        self.transform = transform

    def __getitem__(self, i):
        cropped_image_path = os.path.join(self.data_file + '_preprocessed', self.file[i][:-4] + '_cropped.png')
        cropped_img = Image.open(cropped_image_path).convert('RGB')
        cropped_img = self.transform(cropped_img)

        masked_image_path = os.path.join(self.data_file + '_preprocessed', self.file[i][:-4] + '_masked.png')
        masked_img = Image.open(masked_image_path).convert('RGB')
        masked_img = self.transform(masked_img)
      
        label = self.emo_list.index(self.file[i].split('/')[-2])

        return cropped_img, masked_img, label

    def __len__(self):
        return len(self.file)

