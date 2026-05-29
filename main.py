import os
os.environ['MPLCONFIGDIR'] = "/tmp"
import random
import numpy as np
import torch
import logging

from args import parse_args
from config import (cfg, assert_cfg, dump_cfg,
                             update_out_dir, get_parent_dir)
from dataset import create_dataset,create_loader
from optimizer import create_optimizer, create_scheduler
from model_builder import create_model
from datetime import datetime
from train_live_update import train_live_update
import warnings
from logger import setup_printing, create_logger
from utils import auto_select_device

warnings.filterwarnings("ignore")


def seed_anything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dataset_cfg_setup_live_update(name: str):
    """
    Setup required fields in cfg for the given dataset.
    """
    if name in ['bitcoinotc.csv', 'bitcoinalpha.csv']:
        cfg.dataset.format = 'bitcoin'
        cfg.dataset.edge_dim = 2
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'W'

    elif name in ['CollegeMsg.txt']:
        cfg.dataset.format = 'uci_message'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'W'

    elif name in ['reddit-body.tsv', 'reddit-title.tsv']:
        cfg.dataset.format = 'reddit_hyperlink'
        cfg.dataset.edge_dim = 88
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'W'

    elif name in ['AS-733']:
        cfg.dataset.format = 'as'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'D'

    else:
        raise ValueError(f'No default config for dataset {name}.')


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    auc=[]
    mrr=[]
    f1 = []
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.TTA.lr_feat = args.TTA_lr_feat
    cfg.TTA.epochs = args.TTA_epochs
    
    assert_cfg(cfg)

  
    if args.override_data_dir is not None:
        cfg.dataset.dir = args.override_data_dir
    if args.override_remark is not None:
        cfg.remark = args.override_remark

    torch.set_num_threads(cfg.num_threads)
    out_dir_parent = cfg.out_dir
    cfg.seed = args.seed
    
    seed_anything(cfg.seed)

    update_out_dir(out_dir_parent, args.cfg_file)
    dump_cfg(cfg)
    setup_printing()
    auto_select_device()

    
    if cfg.dataset.format == 'infer':
        dataset_cfg_setup_live_update(cfg.dataset.name)
            
    datasets = create_dataset()

    cfg.dataset.num_nodes = datasets[0][0].num_nodes
    loaders = create_loader(datasets)
    meters = create_logger(datasets, loaders)

    model = create_model(datasets)


    optimizer = create_optimizer(model.parameters())
    scheduler = create_scheduler(optimizer)

    for dataset, name in zip(datasets, ('train', 'validation', 'test')):
        print(f'{name} set: {len(dataset)} graphs.')
        all_edge_time = torch.cat([g.edge_time for g in dataset])
        start = int(torch.min(all_edge_time))
        start = datetime.fromtimestamp(start)
        end = int(torch.max(all_edge_time))
        end = datetime.fromtimestamp(end)
        print(f'\tRange: {start} - {end}')

    if cfg.dataset.negative_sample_weight != 'uniform':
        for dataset in datasets:
            dataset.negative_sample_weight = cfg.dataset.negative_sample_weight
        
        
    main_auc, mean_mrr, mean_f1 = train_live_update(meters, model,  optimizer,  scheduler, datasets=datasets)
    auc.append(main_auc)
    mrr.append(mean_mrr)
    f1.append(mean_f1)
    print(np.mean(auc), np.mean(mrr), np.mean(f1))
    logging.info('average AUC across 3 repeats: {}'.format(np.mean(auc)))
    logging.info('average MRR across 3 repeats: {}'.format(np.mean(mrr)))
    logging.info('average F1 across 3 repeats: {}'.format(np.mean(f1)))
    

