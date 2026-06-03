"""
Data loader for bitcoin datasets.
Mar. 27, 2021
"""
import os
from typing import List, Union
import ipdb
import deepsnap
import numpy as np
import pandas as pd
import torch
from deepsnap.graph import Graph
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder



from config import cfg

def load_single_dataset(dataset_dir: str) -> Graph:
    df_trans = pd.read_csv(dataset_dir, sep=',', header=None, index_col=None)
    df_trans.columns = ['SOURCE', 'TARGET', 'RATING', 'TIME'] #【起始节点ID， 目标节点ID， 评分（边的属性）， 边发生的时间戳】
    # NOTE: 'SOURCE' and 'TARGET' are not consecutive.
    #获取所有唯一出现过的节点 ID 数量，作为图中的节点数 ravel() 是将 SOURCE 和 TARGET 两列展平成一维，再去重
    num_nodes = len(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))

    # bitcoin OTC contains decimal numbers, round them.
    #将时间戳从小数转换为整数再转为浮点型（有些数据可能原来是小数，需要标准化）。
    df_trans['TIME'] = df_trans['TIME'].astype(np.int).astype(np.float)
    #检查数据中是否存在缺失值，若存在则抛出异常。
    assert not np.any(pd.isna(df_trans).values)


    #使用 MinMaxScaler 将原始时间戳缩放到 [0, 2] 区间
    time_scaler = MinMaxScaler((0, 2))
    #增加一列 TimestampScaled 作为归一化后的时间
    df_trans['TimestampScaled'] = time_scaler.fit_transform(
        df_trans['TIME'].values.reshape(-1, 1))

    #边特征是 RATING 和 TimestampScaled 组合而成的二维向量
    edge_feature = torch.Tensor(
        df_trans[['RATING', 'TimestampScaled']].values)  # (E, edge_dim)
    # SOURCE and TARGET IDs are already encoded in the csv file.
    # edge_index = torch.Tensor(
    #     df_trans[['SOURCE', 'TARGET']].values.transpose()).long()  # (2, E)

    #获取所有节点 ID，升序排列。
    node_indices = np.sort(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))
    #用 OrdinalEncoder 把原始 SOURCE 和 TARGET 映射为从 0 到 N-1 的整数索引。
    enc = OrdinalEncoder(categories=[node_indices, node_indices])
    #转置后形状为 [2, E]，符合 PyTorch 图表示规范。
    raw_edges = df_trans[['SOURCE', 'TARGET']].values
    edge_index = enc.fit_transform(raw_edges).transpose()
    edge_index = torch.LongTensor(edge_index)

    # num_nodes = torch.max(edge_index) + 1
    # Use dummy node features.
    #为每个节点分配一个 dummy 特征（全为1的标量向量），维度为 (N, 200)
    node_feature = torch.ones(num_nodes, 200).float()

    #原始时间戳仍保留，作为动态图建模所需的 edge_time。
    edge_time = torch.FloatTensor(df_trans['TIME'].values)

    if cfg.train.mode in ['baseline', 'baseline_v2', 'live_update_fixed_split']:
        edge_feature = torch.cat((edge_feature, edge_feature.clone()), dim=0)
        reversed_idx = torch.stack([edge_index[1], edge_index[0]]).clone()
        edge_index = torch.cat((edge_index, reversed_idx), dim=1)
        edge_time = torch.cat((edge_time, edge_time.clone())) #加入反向边

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )
    return graph


def make_graph_snapshot(g_all: Graph, snapshot_freq: str) -> List[Graph]:
    t = g_all.edge_time.numpy().astype(np.int64) #从图中提取边的时间戳 edge_time，并转为 numpy int64 类型（单位为秒）
    snapshot_freq = snapshot_freq.upper()#确保传入的 snapshot_freq 是大写（标准化处理）。

    period_split = pd.DataFrame( #新建一个 DataFrame
        {'Timestamp': t, #'Timestamp'：原始秒级时间戳；
         'TransactionTime': pd.to_datetime(t, unit='s')}, #'TransactionTime'：将其转换为 datetime 类型，方便后续按天/周/月等时间单位划分。
        index=range(len(g_all.edge_time)))

    freq_map = {'D': '%j',  # day of year. %j: 一年中的第几天（001-366）；
                'W': '%W',  # week of year. %W: 一年中的第几周（00-53）；
                'M': '%m'  # month of year. %m: 月份（01-12）
                }

    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int) #提取时间戳的年份部分（例如 2023）。

    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int) #结合前面定义的 freq_map，按选择的粒度提取时间片段编号

    #使用 groupby 按 (Year, SubYearFlag) 分组，得到每个时间段对应的边的索引列表 结果如：{(2023, 1): [0,1,2], (2023, 2): [3,4,5], ...} #【0,1,2】在那个时间错下对应的边索引
    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices 
    

    periods = sorted(list(period2id.keys())) #按时间顺序排列所有 (Year, SubYearFlag)。
    
    snapshot_list = list() #准备一个空列表，用于存放每个快照图。

    for p in periods: #遍历每一个时间段 p；
        # unique IDs of edges in this period.
        period_members = period2id[p] #period_members: 属于该时间段的边的索引；
        
        assert np.all(period_members == np.unique(period_members)) #用断言确保这些索引是唯一的（无重复边）。

        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed
        )#Graph(directed=[1], edge_feature=[32, 2], edge_index=[2, 32], edge_label_index=[2, 32], edge_time=[32], node_feature=[5881, 1], node_label_index=[5881])
        #ipdb.set_trace()
        snapshot_list.append(g_incr)

    snapshot_list.sort(key=lambda x: torch.min(x.edge_time))#有可能 period2id 中时间顺序不严谨，因此再次确保快照图按边时间排序。

    return snapshot_list


def split_by_seconds(g_all, freq_sec: int):
    # Split the entire graph into snapshots.
    split_criterion = g_all.edge_time // freq_sec
    groups = torch.sort(torch.unique(split_criterion))[0]
    snapshot_list = list()
    for t in groups:
        period_members = (split_criterion == t)
        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed
        )
        snapshot_list.append(g_incr)
    return snapshot_list


def load_generic(dataset_dir: str,
                 snapshot: bool = True,
                 snapshot_freq: str = None
                 ) -> Union[deepsnap.graph.Graph,
                            List[deepsnap.graph.Graph]]:
    g_all = load_single_dataset(dataset_dir)
    if not snapshot:
        return g_all

    if snapshot_freq.upper() not in ['D', 'W', 'M']:
        # format: '1200000s'
        # assume split by seconds (timestamp) as in EvolveGCN paper.
        freq = int(snapshot_freq.strip('s'))
        snapshot_list = split_by_seconds(g_all, freq)
    else:
        snapshot_list = make_graph_snapshot(g_all, snapshot_freq)
    num_nodes = g_all.edge_index.max() + 1

    for g_snapshot in snapshot_list:
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_degree_existing = torch.zeros(num_nodes)

    # check snapshots ordering.
    prev_end = -1
    for g in snapshot_list:
        start, end = torch.min(g.edge_time), torch.max(g.edge_time)
        assert prev_end < start <= end
        prev_end = end

    return snapshot_list


def load_btc_dataset(format, name, dataset_dir):
    if format == 'bitcoin':
        graphs = load_generic(os.path.join(dataset_dir, name),
                              snapshot=cfg.transaction.snapshot,
                              snapshot_freq=cfg.transaction.snapshot_freq)
        if cfg.dataset.split_method == 'chronological_temporal':
            return graphs
        else:
            # The default split (80-10-10) requires at least 10 edges each
            # snapshot.
            filtered_graphs = list()
            for g in graphs:
                if g.num_edges >= 10:
                    filtered_graphs.append(g)
            return filtered_graphs


