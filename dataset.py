
import networkx as nx
import time
import logging
import pickle

from deepsnap.dataset import GraphDataset
import torch
from torch.utils.data import DataLoader

from torch_geometric.datasets import *
import torch_geometric.transforms as T

import networks.feature_augment as preprocess
from transform import (ego_nets, remove_node_feature,
                                       edge_nets, path_len)
from config import cfg
from deepsnap.batch import Batch
import ipdb
import pdb
from dataset_btc import load_btc_dataset
from dataset_CM import load_CM_dataset
from dataset_reddit import load_reddit_dataset
from dataset_AS import load_AS_dataset
from config import cfg

def load_dataset():
    '''
    load raw datasets.
    :return: a list of networkx/deepsnap graphs, plus additional info if needed
    '''
    format = cfg.dataset.format
    name = cfg.dataset.name
    # dataset_dir = '{}/{}'.format(cfg.dataset.dir, name)
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    if name in ['bitcoinotc.csv', 'bitcoinalpha.csv']:
        graphs = load_btc_dataset(format, name, dataset_dir)

    elif name in ['CollegeMsg.txt']:
        graphs = load_CM_dataset(format, name, dataset_dir)

    elif name in ['reddit-body.tsv', 'reddit-title.tsv']:
        graphs = load_reddit_dataset(format, name, dataset_dir)

    elif name in ['AS-733']:
        graphs = load_AS_dataset(format, name, dataset_dir)

    else:
        raise ValueError(f'No default config for dataset {name}.')

    return graphs


def filter_graphs():
    '''
    Filter graphs by the min number of nodes
    :return: min number of nodes
    '''
    if cfg.dataset.task == 'graph':
        min_node = 0
    else:
        min_node = 5
    return min_node


def transform_before_split(dataset):
    '''
    Dataset transformation before train/val/test split
    :param dataset: A DeepSNAP dataset object
    :return: A transformed DeepSNAP dataset object
    '''
    if cfg.dataset.remove_feature:
        dataset.apply_transform(remove_node_feature,
                                update_graph=True, update_tensor=False)
        print('1')
    augmentation = preprocess.FeatureAugment()
    actual_feat_dims, actual_label_dim = augmentation.augment(dataset)
    if cfg.dataset.augment_label:
        dataset.apply_transform(preprocess._replace_label,
                                update_graph=True, update_tensor=False)
        print('2')
    # Update augmented feature/label dims by real dims (user specified dims
    # may not be realized)
    cfg.dataset.augment_feature_dims = actual_feat_dims
    if cfg.dataset.augment_label:
        cfg.dataset.augment_label_dims = actual_label_dim
        print('3')

    # Temporary for ID-GNN path prediction task
    if cfg.dataset.task == 'edge' and 'id' in cfg.gnn.layer_type:
        dataset.apply_transform(path_len, update_graph=False,
                                update_tensor=False)
        print('4')

    return dataset


def transform_after_split(datasets):
    '''
    Dataset transformation after train/val/test split
    :param dataset: A list of DeepSNAP dataset objects
    :return: A list of transformed DeepSNAP dataset objects
    '''
    if cfg.dataset.transform == 'ego':
        for split_dataset in datasets:
            split_dataset.apply_transform(ego_nets,
                                          radius=cfg.gnn.layers_mp,
                                          update_tensor=True,
                                          update_graph=False)
        print('5')
    elif cfg.dataset.transform == 'edge':
        for split_dataset in datasets:
            split_dataset.apply_transform(edge_nets,
                                          radius=cfg.gnn.layers_mp,
                                          update_tensor=True,
                                          update_graph=False)
            split_dataset.task = 'node'
        print('6')
        cfg.dataset.task = 'node'
    return datasets


def create_dataset():
    ## Load dataset
    time1 = time.time()
    
    graphs = load_dataset()

    ## Filter graphs
    time2 = time.time()
    min_node = filter_graphs()

    ## Create whole dataset
    dataset = GraphDataset(
        graphs,
        task=cfg.dataset.task,
        edge_train_mode=cfg.dataset.edge_train_mode,
        edge_message_ratio=cfg.dataset.edge_message_ratio,
        edge_negative_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
        minimum_node_per_graph=min_node)
    
    
    
    #print(dataset[1].edge_label_index)

    ## Transform the whole dataset
    dataset = transform_before_split(dataset)

    ## Split dataset
    time3 = time.time()
    # Use custom data splits

    datasets = dataset.split(
        transductive=cfg.dataset.transductive,
        split_ratio=cfg.dataset.split,
        shuffle=cfg.dataset.shuffle) # 返回python列表 datasets[0] → 训练集（training set） datasets[1] → 验证集（validation set） datasets[2] → 测试集（test set）
    #print('live updata split')
    ## Transform each split dataset
    time4 = time.time()
    datasets = transform_after_split(datasets)


    time5 = time.time()
    logging.info('Load: {:.4}s, Before split: {:.4}s, '
                 'Split: {:.4}s, After split: {:.4}s'.format(
        time2 - time1, time3 - time2, time4 - time3, time5 - time4))
    return datasets



def create_loader(datasets):
    loader_train = DataLoader(datasets[0], collate_fn=Batch.collate(),
                              batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=False)

    loaders = [loader_train]
    for i in range(1, len(datasets)):
        loaders.append(DataLoader(datasets[i], collate_fn=Batch.collate(),
                                  batch_size=cfg.train.batch_size,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False))

    return loaders
