
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from Attention_Classification.attention_config import attention_config as config


import torch
import pickle

import os
from os.path import join

def get_inference_dataloader(processor, data, mode='test'):
    examples = processor.create_examples(data)
    features = processor.convert_examples_to_features(examples)
    input_ids, labels, word_lengths, sent_lengths = processor.get_features(features, mode)
    dataset = TensorDataset(input_ids, word_lengths, sent_lengths)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=config['batch_size'])
    return loader

def get_dataloader(processor, mode):

    examples = processor.get_examples(mode)
    features = processor.convert_examples_to_features(examples)
    input_ids, labels, word_lengths, sent_lengths = processor.get_features(features, mode)

    if mode in ['train', 'valid']:
        dataset = TensorDataset(input_ids, labels, word_lengths, sent_lengths)
        sampler = RandomSampler(dataset)
    elif mode in ['test']:
        dataset = TensorDataset(input_ids, word_lengths, sent_lengths)
        sampler = SequentialSampler(dataset)

    loader = DataLoader(dataset, sampler=sampler, batch_size=config['batch_size'])
    return loader


def get_device_n_gpus(rank=-1):

    if torch.cuda.is_available():
        if rank == -1:
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()
        elif rank == -2:
            device = torch.device("cpu")
            n_gpu = 0
        else:
            device = torch.device("cuda", rank)
            n_gpu = 1
    else:
        device = torch.device("cpu")
        n_gpu = 0

    return  device, n_gpu

def save_model(model, n_gpu, model_dir, name):
    path_to_model = join(model_dir, name)
    if n_gpu >1:
        torch.save(model.module.state_dict(), path_to_model)
    else:
        torch.save(model.state_dict(), path_to_model)


def save_stats(stats, stats_dir, name):
    path_to_stats = join(stats_dir, name)
    with open(path_to_stats, 'wb') as f:
        pickle.dump(stats, f)

def load_stats(stats_dir, name):
    path_to_stats = join(stats_dir, name)
    # if not os.path.exists(path_to_stats):
    #     stats = Stats(config)
    #     return stats
    with open(path_to_stats, 'rb') as f:
        stats = pickle.load(f)
    return stats


def create_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_folders():
    create_folder(config['model_dir'])
    create_folder(config['stats_dir'])
    create_folder(config['results_dir'])


