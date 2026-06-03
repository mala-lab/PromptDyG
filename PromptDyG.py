from torch.nn.parameter import Parameter
from copy import deepcopy
from config import cfg
from networks import train_utils
from loss import compute_loss
from sklearn.metrics import roc_auc_score,f1_score
from tqdm import tqdm
from torch_geometric.utils import dropout_adj
from utils import get_task_batch
import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
#from plot_img import plot_similarity_distribution,visualize_embedding_shift

class PromptDyG():
    def __init__(self, config):
        self.device = config.device
        self.config = config
        self.epochs = cfg.TTA.epochs

    def optim_prompt(self, model, datasets, t, prev_node_states):
        config = self.config

        batch = get_task_batch(datasets[2], t, t + 1, prev_node_states).clone()
        self.batch = batch
        nnodes = batch.node_feature.shape[0]
        d = batch.node_feature.shape[1]

        prompt_feat = Parameter(torch.FloatTensor(nnodes, d).to(self.device))
        prompt_feat.data.fill_(1e-7)
        self.prompt_feat = prompt_feat
        
        self.optimizer_feat = torch.optim.Adam([self.prompt_feat], lr=cfg.TTA.lr_feat)

        #self.model = model
        for param in model.parameters():
            param.requires_grad = False
        model.eval() # should set to eval

        self.edge_index, self.feat = batch.edge_index, batch.node_feature

        best_loss = float('inf')
        best_auroc = -float('inf')
        #best_f1 = -float('inf')
        best_state_dict = None
        best_epoch = -1
        best_mrr, best_auroc, best_f1 = -1, -1, -1


        for it in tqdm(range(self.epochs)):
            
            self.optimizer_feat.zero_grad()
            loss = self.test_time_loss(model)

            loss.backward()

            self.optimizer_feat.step()
    
            with torch.no_grad():
                batch_new = deepcopy(self.batch)
                batch_new.node_feature = self.feat + self.prompt_feat
                pred, true = model(batch_new)
                mrr, auroc, f1 = self.evaluate_single(pred, true, model, datasets[2], t, prev_node_states)
                
            
            
            if loss < best_loss:
                best_loss = loss
                best_mrr = mrr
                best_auroc = auroc
                best_f1 = f1
                best_epoch = it
                #best_state_dict = deepcopy(model.state_dict())             
                print("\033[42;37mNew best model at epoch {}, mrr={:.4f}\033[0m".format(it, best_mrr))
            

                
        print("\033[42;37mbest_mrr:{}, best_auroc:{}, best_f1:{}\033[0m".format(
            best_mrr, best_auroc, best_f1))

        
        return best_mrr, best_auroc, best_f1
        



    def evaluate_single(self, pred, true, model, dataset, t, prev_node_states):
        _, pred_score = compute_loss(pred, true)

        auroc = roc_auc_score(y_true=true.detach().cpu().numpy(), y_score=pred_score.detach().cpu().numpy())

        pred_label = torch.zeros(len(pred_score))
        pred_label[pred_score >= 0.5] = 1.0
        accuracy = np.mean(true.detach().cpu().numpy() == pred_label.numpy())
        f1 = f1_score(true.detach().cpu().numpy(), pred_label.numpy())

        mrr_batch = get_task_batch(dataset, t, t + 1,
                               prev_node_states).clone()
        
        mrr_batch.node_feature = mrr_batch.node_feature + self.prompt_feat
        

        mrr, rck1, rck3, rck10 = train_utils.report_rank_based_eval(
                        mrr_batch, model, num_neg_per_node=cfg.experimental.rank_eval_multiplier)
        
        return mrr, auroc, f1


    def test_time_loss(self, model, mode='train', e_margin=math.log(1000)/2-1):
        loss =0

        batch_new = deepcopy(self.batch)
        if hasattr(self, 'prompt_feat'):           
            batch_new.node_feature = self.feat + self.prompt_feat
        else:
            batch_new.node_feature = self.feat
        batch_size = 1000
        output = model.get_embed(batch_new)

        sampled = np.random.permutation(np.arange(len(output))[: batch_size])
        loss += softmax_entropy(output[sampled]).mean(0)
        
        return loss



@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **logits**."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **log_softmax**."""
    return -(x * torch.log(x+1e-15)).sum(1)


