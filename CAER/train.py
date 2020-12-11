import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.backends.cudnn as cudnn
import numpy as np
import random
from torchvision import models
from data.datamgr import SimpleDataManager
from model import CAERnet, OURnet, Context

if __name__ == '__main__':
    base_lr = 1e-4
    batch_size = 32
    power = 0.9
    num_epochs = 300
    save_epoch = 10
    num_workers = 4
    image_size = 224
    seed = 1339

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    checkpoint_dir = './checkpoint/'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_datamgr = SimpleDataManager(image_size=image_size, batch_size=batch_size, num_workers=num_workers)
    train_loader = train_datamgr.get_data_loader(data_path='PATH/TO/DATASET/CAER-S', load_set='train', aug=True)

    test_datamgr = SimpleDataManager(image_size=image_size, batch_size=batch_size, num_workers=num_workers)
    test_loader = test_datamgr.get_data_loader(data_path='PATH/TO/DATASET/CAER-S', load_set='test', aug=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = OURnet()
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    CE_loss = nn.CrossEntropyLoss()

    
    optimizer = optim.Adam([{'params': model.face_features.parameters(), 'lr': base_lr},
                           {'params': model.context_features.parameters(), 'lr': base_lr},
                           {'params': model.multi_transformer.parameters(), 'lr': base_lr * 0.1}],
                          lr=base_lr)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (x1, x2, y) in enumerate(train_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x1, x2)
            loss = CE_loss(output, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch + 1, num_epochs,
                                                                             i + 1, len(train_loader),
                                                                             running_loss / 100))
                running_loss = 0.0

        if epoch % save_epoch == save_epoch - 1:
            save_file = os.path.join(checkpoint_dir, str(epoch + 1) + '.pth')
            torch.save(model.state_dict(), save_file)

    """ Evaluation """
    for j in range(int(num_epochs / save_epoch)):
        num_model = save_epoch * (j + 1)

        save_file = os.path.join(checkpoint_dir, str(num_model) + '.pth')

        model = OURnet()
        model = model.to(device)

        loaded_params = torch.load(save_file)
        new_params = model.state_dict().copy()
        for i in loaded_params:
            i_parts = i.split('.')
            if i_parts[0] == 'module':
                new_params['.'.join(i_parts[1:])] = loaded_params[i]
            else:
                new_params[i] = loaded_params[i]
        model.load_state_dict(new_params)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.eval()

        correct = 0.0
        count = 0
        with torch.no_grad():
            for k, (x1, x2, y) in enumerate(train_loader):
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                outputs = model(x1, x2)

                _, predicted = torch.max(outputs, 1)
                c = (predicted == y).squeeze()

                correct += c.sum()
                count += x1.shape[0]
        print('Train Accuracy({:d}) {:.2f}%'.format(num_model, correct * 100 / count))

        correct = 0.0
        count = 0
        with torch.no_grad():
            for k, (x1, x2, y) in enumerate(test_loader):
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                outputs = model(x1, x2)

                _, predicted = torch.max(outputs, 1)
                c = (predicted == y).squeeze()

                correct += c.sum()
                count += x1.shape[0]
        print('Test Accuracy({:d}) {:.2f}%'.format(num_model, correct * 100 / count))
