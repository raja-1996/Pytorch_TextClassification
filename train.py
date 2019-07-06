
import torch
import torch.nn as nn
import torch.optim as optim

from Attention_Classification.utils import get_dataloader, get_device_n_gpus, save_model, save_stats, create_folders, load_stats
from Attention_Classification.HierarchicalAttention import HierarchicalAttention
from Attention_Classification.attention_config import attention_config as config
from Attention_Classification.Stats import Stats
from Attention_Classification.DataProcessor import Dataprocessor

import pandas as pd
import os
from os.path import join
# Args


#Create folders
create_folders()

# Dataset and Dataloader

processor = Dataprocessor(config)
train_loader = get_dataloader(processor, 'train')
dev_loader = get_dataloader(processor, 'valid')
test_loader = get_dataloader(processor, 'test')

labels_set = processor.get_labels()
num_classes = len(labels_set)
config['num_classes'] = num_classes

print('Data Loaded !!!!')

#device

device, n_gpu = get_device_n_gpus(config['rank'])
config['device'] = device
config['n_gpu'] = n_gpu

print('Device selected {} and n_gpus {}'.format(device, n_gpu))

# model creation

model = HierarchicalAttention(config)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=1e-4,
                       betas=[0.9, 0.999],
                       eps=1e-9,
                       weight_decay=1e-5)
#stats
path_to_stats = join(config['stats_dir'], config['stats_name'])
if not os.path.exists(path_to_stats):
    stats = Stats(config)
else:
    print('Loading Stats from Disk')
    stats = load_stats(config['stats_dir'], config['stats_name'])

test_df = pd.DataFrame()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total params', total_params)

#epoch loop
start_epoch = stats.last_epoch
try:
    num_batches = len(train_loader)
    num_epochs = config['num_epochs']
    for epoch in range(start_epoch+1, num_epochs):

        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):

            batch = [each.to(device) for each in batch]

            input_ids, labels, word_lengths, sent_lengths = batch
            scores, word_attn_scores, sent_attn_scores = model(input_ids, word_lengths, sent_lengths)
            # print(scores)
            loss = criterion(scores, labels)
            if n_gpu > 1:
                loss = loss.mean()
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            print('Loss at Batch {}/{}: {}'.format(i, num_batches, loss), end='\r')
        train_loss = train_loss/num_batches
        print('Training Loss after Epoch {}/{}: {}'.format(epoch, num_epochs, train_loss))
        
        model.eval()
        dev_loss = 0.0
        for i, batch in enumerate(dev_loader):
            batch = [each.to(device) for each in batch]

            input_ids, labels, word_lengths, sent_lengths = batch
            scores, word_attn_scores, sent_attn_scores = model(input_ids, word_lengths, sent_lengths)

            loss = criterion(scores, labels)
            if n_gpu > 1:
                loss = loss.mean()
            dev_loss += loss.item()
        dev_loss = dev_loss/len(dev_loader)
        print('Validation Loss after Epoch {}/{}: {}'.format(epoch, num_epochs, dev_loss))

        classes = []
        for i, batch in enumerate(test_loader):

            batch = [each.to(device) for each in batch]
            input_ids, word_lengths, sent_lengths = batch

            scores, word_attn_scores, sent_attn_scores = model(input_ids, word_lengths, sent_lengths)
            idxs = torch.argmax(scores, dim=-1)
            classes_ = [labels_set[idx] for idx in idxs.tolist()]
            classes.extend(classes_)

        test_df['test_{}'.format(epoch)] = classes
        print()
        stats.on_epoch_end(model, epoch, train_loss, dev_loss)
        if stats.stop_training:
            stats.on_train_end()
            break
except KeyboardInterrupt:
    print('KeyboardInterrupt')


# Save Model
print('Saving Model !!!!')
save_model(model, n_gpu, config['model_dir'], config['model_name'])
save_stats(stats, config['stats_dir'], config['stats_name'])
test_df.to_csv(join(config['results_dir'], 'test_results.csv'), index=False)
stats.plot_loss()