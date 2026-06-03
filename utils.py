import torch
import subprocess
import numpy as np
import pdb
from config import cfg
import logging
import deepsnap
from typing import Dict, List, Optional, Tuple
from networks import train_utils

def node_degree(edge_index, n=None, mode='in'):
    if mode == 'in':
        index = edge_index[0, :]
    elif mode == 'out':
        index = edge_index[1, :]
    else:
        index = edge_index.flatten()
    n = edge_index.max() + 1 if n is None else n
    degree = torch.zeros(n)
    ones = torch.ones(index.shape[0])
    return degree.scatter_add_(0, index, ones)

@torch.no_grad()
def get_task_batch(dataset: deepsnap.dataset.GraphDataset,
                   today: int, tomorrow: int,
                   prev_node_states: Optional[Dict[str, List[torch.Tensor]]]
                   ) -> deepsnap.graph.Graph:
    """
    Construct batch required for the task (today, tomorrow). As defined in
    batch's get_item method (used to get edge_label and get_label_index),
    edge_label and edge_label_index returned would be different everytime
    get_task_batch() is called.

    Moreover, copy node-memories (node_states and node_cells) to the batch.
    """
    assert today <= tomorrow < len(dataset)
    # Get edges for message passing and prediction task.
    batch = dataset[today].clone()
    batch.edge_label = dataset[tomorrow].edge_label.clone()
    batch.edge_label_index = dataset[tomorrow].edge_label_index.clone()

    # Copy previous memory to the batch.
    if prev_node_states is not None:
        for key, val in prev_node_states.items():
            copied = [x.detach().clone() for x in val]
            setattr(batch, key, copied)

    batch = train_utils.move_batch_to_device(batch, cfg.device)
    return batch






def get_gpu_memory_map():
    '''Get the current gpu usage.'''
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    return gpu_memory


def auto_select_device(memory_max=8000, memory_bias=200, strategy='random'):
    '''Auto select GPU device'''
    if cfg.device != 'cpu' and torch.cuda.is_available():
        if cfg.device == 'auto':
            memory_raw = get_gpu_memory_map()
            if strategy == 'greedy' or np.all(memory_raw > memory_max):
                cuda = np.argmin(memory_raw)
                logging.info('GPU Mem: {}'.format(memory_raw))
                logging.info(
                    'Greedy select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))
            elif strategy == 'random':
                memory = 1 / (memory_raw + memory_bias)
                memory[memory_raw > memory_max] = 0
                gpu_prob = memory / memory.sum()
                np.random.seed()
                cuda = np.random.choice(len(gpu_prob), p=gpu_prob)
                np.random.seed(cfg.seed)
                logging.info('GPU Mem: {}'.format(memory_raw))
                logging.info('GPU Prob: {}'.format(gpu_prob.round(2)))
                logging.info(
                    'Random select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))

            cfg.device = 'cuda:{}'.format(cuda)
    else:
        cfg.device = 'cpu'
