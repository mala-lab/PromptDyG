import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
import torch.nn as nn
import deepsnap
import register as register
from config import cfg
# Used for the OGB Encoders
full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


######## Feature Encoders #########
class IntegerFeatureEncoder(torch.nn.Module):
    """
        Provides an encoder for integer node features

        Parameters:
        num_classes - the number of classes for the embedding mapping to learn
    """

    def __init__(self, emb_dim, num_classes=None):
        super(IntegerFeatureEncoder, self).__init__()

        self.encoder = torch.nn.Embedding(num_classes, emb_dim)
        torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        batch.node_feature = self.encoder(batch.node_feature[:, 0])

        return batch


class SingleAtomEncoder(torch.nn.Module):
    """
        Only encode the first dimension of atom integer features.
        This feature encodes just the atom type

        Parameters:
        num_classes: Not used!
    """

    def __init__(self, emb_dim, num_classes=None):
        super(SingleAtomEncoder, self).__init__()

        num_atom_types = full_atom_feature_dims[0]
        self.atom_type_embedding = torch.nn.Embedding(num_atom_types, emb_dim)
        torch.nn.init.xavier_uniform_(self.atom_type_embedding.weight.data)

    def forward(self, batch):
        batch.node_feature = self.atom_type_embedding(batch.node_feature[:, 0])

        return batch


class AtomEncoder(torch.nn.Module):
    """
        The complete Atom Encoder used in OGB dataset

        Parameters:
        num_classes: Not used!
    """

    def __init__(self, emb_dim, num_classes=None):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, batch):
        encoded_features = 0
        for i in range(batch.node_feature.shape[1]):
            encoded_features += self.atom_embedding_list[i](
                batch.node_feature[:, i])

        batch.node_feature = encoded_features
        return batch


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, batch):
        bond_embedding = 0
        for i in range(batch.edge_feature.shape[1]):
            bond_embedding += self.bond_embedding_list[i](
                batch.edge_feature[:, i])

        batch.edge_feature = bond_embedding
        return batch

class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim: int):
        # emb_dim is not used here.
        super(LinearEdgeEncoder, self).__init__()
        # For consistency, the edge features will be map to this dimension
        # on the BSI dataset.
        expected_dim = cfg.transaction.feature_amount_dim \
            + cfg.transaction.feature_time_dim
        
        self.linear = nn.Linear(cfg.dataset.edge_dim, expected_dim)
        cfg.dataset.edge_dim = expected_dim

    def forward(self, batch: deepsnap.batch.Batch) -> deepsnap.batch.Batch:
        batch.edge_feature = self.linear(batch.edge_feature)
        return batch



class TransactionEdgeEncoder(torch.nn.Module):
    r"""A module that encodes edge features in the transaction graph.

    Example:
        TransactionEdgeEncoder(
          (embedding_list): ModuleList(
            (0): Embedding(50, 32)  # The first integral edge feature has 50 unique values.
                # convert this integral feature to 32 dimensional embedding.
            (1): Embedding(8, 32)
            (2): Embedding(252, 32)
            (3): Embedding(252, 32)
          )
          (linear_amount): Linear(in_features=1, out_features=64, bias=True)
          (linear_time): Linear(in_features=1, out_features=64, bias=True)
        )

        Initial edge feature dimension = 6
        Final edge embedding dimension = 32 + 32 + 32 + 32 + 64 + 64 = 256
    """
    def __init__(self, emb_dim: int):
        # emb_dim is not used here.
        super(TransactionEdgeEncoder, self).__init__()

        self.embedding_list = torch.nn.ModuleList()
        # Note: feature_edge_int_num[i] = len(torch.unique(graph.edge_feature[:, i]))
        # where i-th edge features are integral.
        for num in cfg.transaction.feature_edge_int_num:
            emb = torch.nn.Embedding(num, cfg.transaction.feature_int_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.embedding_list.append(emb)

        # Embed non-integral features.
        self.linear_amount = nn.Linear(1, cfg.transaction.feature_amount_dim)
        self.linear_time = nn.Linear(1, cfg.transaction.feature_time_dim)
        # update edge_dim
        cfg.dataset.edge_dim = len(cfg.transaction.feature_edge_int_num) \
                               * cfg.transaction.feature_int_dim \
                               + cfg.transaction.feature_amount_dim \
                               + cfg.transaction.feature_time_dim

    def forward(self, batch: deepsnap.batch.Batch) -> deepsnap.batch.Batch:
        edge_embedding = []
        for i in range(len(self.embedding_list)):
            edge_embedding.append(
                self.embedding_list[i](batch.edge_feature[:, i].long())
            )
        # By default, edge_feature[:, -2] contains edge amount,
        # edge_feature[:, -1] contains edge time.
        edge_embedding.append(
            self.linear_amount(batch.edge_feature[:, -2].view(-1, 1))
        )
        edge_embedding.append(
            self.linear_time(batch.edge_feature[:, -1].view(-1, 1))
        )
        batch.edge_feature = torch.cat(edge_embedding, dim=1)
        return batch



class TransactionNodeEncoder(torch.nn.Module):
    r"""A module that encodes node features in the transaction graph.

    Parameters:
        num_classes - the number of classes for the embedding mapping to learn

    Example:
        3 unique values for the first integral node feature.
        3 unique values for the second integral node feature.

        cfg.transaction.feature_node_int_num = [3, 3]
        cfg.transaction.feature_int_dim = 32

        TransactionNodeEncoder(
          (embedding_list): ModuleList(
            (0): Embedding(3, 32)  # embed the first node feature to 32-dimensional space.
            (1): Embedding(3, 32)  # embed the second node feature to 32-dimensional space.
          )
        )

        Initial node feature dimension = 2
        Final node embedding dimension = 32 + 32 = 256
    """

    def __init__(self, emb_dim: int, num_classes=None):
        super(TransactionNodeEncoder, self).__init__()
        self.embedding_list = torch.nn.ModuleList()
        for i, num in enumerate(cfg.transaction.feature_node_int_num):
            emb = torch.nn.Embedding(num, cfg.transaction.feature_int_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.embedding_list.append(emb)
        # update encoder_dim
        cfg.dataset.encoder_dim = len(cfg.transaction.feature_node_int_num) \
                                    * cfg.transaction.feature_int_dim

    def forward(self, batch: deepsnap.batch.Batch) -> deepsnap.batch.Batch:
        node_embedding = []
        for i in range(len(self.embedding_list)):
            node_embedding.append(
                self.embedding_list[i](batch.node_feature[:, i].long())
            )
        batch.node_feature = torch.cat(node_embedding, dim=1)
        return batch



node_encoder_dict = {
    'Integer': IntegerFeatureEncoder,
    'SingleAtom': SingleAtomEncoder,
    'Atom': AtomEncoder,
    'roland': TransactionNodeEncoder,

}

node_encoder_dict = {**register.node_encoder_dict, **node_encoder_dict}

edge_encoder_dict = {
    'Bond': BondEncoder,
    'roland_general': LinearEdgeEncoder,
    'roland': TransactionEdgeEncoder,
}

edge_encoder_dict = {**register.edge_encoder_dict, **edge_encoder_dict}


