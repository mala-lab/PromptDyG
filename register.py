import torch
import torch.nn as nn



def register(key, module, module_dict):
    if key in module_dict:
        raise KeyError('Key {} is already pre-defined.'.format(key))
    else:
        module_dict[key] = module

config_dict = {}
def register_config(key, module):
    register(key, module, config_dict)

layer_dict = {}
def register_layer(key, module):
    register(key, module, layer_dict)

act_dict = {}
def register_act(key, module):
    register(key, module, act_dict)

stage_dict = {}
def register_stage(key, module):
    register(key, module, stage_dict)

head_dict = {}
def register_head(key, module):
    register(key, module, head_dict)

pooling_dict = {}
def register_pooling(key, module):
    register(key, module, pooling_dict)

feature_augment_dict = {}
def register_feature_augment(key, module):
    register(key, module, feature_augment_dict)

node_encoder_dict = {}
def register_node_encoder(key, module):
    register(key, module, node_encoder_dict)

edge_encoder_dict = {}
def register_edge_encoder(key, module):
    register(key, module, edge_encoder_dict)
