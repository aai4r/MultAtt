from __future__ import print_function
import torch.backends.cudnn as cudnn
from model import Ours
from torch.utils.data import DataLoader
import os
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from dataloader import MOSI_Dataset, multi_collate

seed = 524

def acc_and_f1(preds, labels):
    acc = (preds == labels).sum() * 100 / len(preds)
    f1 = f1_score(y_true=labels, y_pred=preds) * 100
    return acc, f1

def adjust_learning_rate(optimizer, base_lr, i_iter, max_iter, power):
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr

def main(options):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Ours()
    model.to(device)
    print("Model initialized")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([{'params': model.Transformer1.parameters(), 'lr': 1e-4}], lr=1e-4)

    # setup training
    batch_sz = options['batch_size']
    epochs = options['epochs']
    save_epoch = options['save_epoch']
    model_path = options['model_path']
    finetuned = options['finetuned']
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    train_data = MOSI_Dataset('PATH/TO/DATASET/MOSI', 'train', finetuned=finetuned)
    train_loader = DataLoader(train_data, batch_size=batch_sz, shuffle=True, num_workers=4, collate_fn=multi_collate)
    train_test_loader = DataLoader(train_data, batch_size=1282, shuffle=False, num_workers=4, collate_fn=multi_collate)

    test_data = MOSI_Dataset('PATH/TO/DATASET/MOSI', 'test', finetuned=finetuned)
    test_loader = DataLoader(test_data, batch_size=684, shuffle=False, num_workers=4, collate_fn=multi_collate)

    for e in range(epochs):
        model.train()
        adjust_learning_rate(optimizer, 1e-4, e, epochs, power=0.9)
        train_loss = 0.0
        for (i, batch) in enumerate(train_loader):
            l, a, v, y = batch
            l, a, v, y = l.to(device), a.to(device), v.to(device), y.to(device)
            y = (y > 0).type(torch.long)
        
            optimizer.zero_grad()

            output = model(l, a, v)
            loss = criterion(output, y)
  
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % 10 == 9 and i != 0:
                print("Epoch {}: {} / {}".format(e + 1, i + 1, len(train_loader)))
        print("Epoch {0} complete! Average Training loss: {1:.4f}".format(e + 1, train_loss / len(train_loader)))

        # Terminate the training process if run into NaN
        if np.isnan(train_loss):
            print("Training got into NaN values...\n\n")
            break

        # if e % save_epoch == save_epoch - 1:
        #     save_file = os.path.join(model_path, str(e + 1) + '.pth')
        #     torch.save(model.state_dict(), save_file)

        model.eval()
        for batch in train_test_loader:
            l, a, v, y = batch
            l, a, v, y = l.to(device), a.to(device), v.to(device), y.to(device)
            y = (y > 0).type(torch.long)
   
            output_test = model(l, a, v)
            preds = torch.argmax(output_test, 1)
      
        y = y.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()

        acc, f1 = acc_and_f1(preds, y)
        print("Epoch {0} | Train. | Acc: {1:.2f}, F1: {2:.2f}".format(e + 1, acc, f1))
    
        for batch in test_loader:
            l, a, v, y = batch
            l, a, v, y = l.to(device), a.to(device), v.to(device), y.to(device)
            y = (y > 0).type(torch.long)
 
            output_test = model(l, a, v)
            preds = torch.argmax(output_test, 1)
     
        y = y.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()

        acc, f1 = acc_and_f1(preds, y)
        print("Epoch {0} | Test. | Acc: {1:.2f}, F1: {2:.2f}".format(e + 1, acc, f1))
        print("=" * 50)


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=100)
    OPTIONS.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    OPTIONS.add_argument('--save_epoch', dest='save_epoch', type=int, default=1)
    OPTIONS.add_argument('--finetuned', dest='finetuned', type=bool, default=True)
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--model_path', dest='model_path', type=str, default='./models')
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
