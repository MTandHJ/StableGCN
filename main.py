

from typing import Dict, Optional, Union

import torch, os, math
import torch.nn as nn
import torch_geometric.transforms as T
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, HeteroData 
from torch_geometric.nn import LGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_scipy_sparse_matrix

import freerec
from freerec.data.postprocessing import RandomIDs, OrderedIDs
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BPRLoss
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, ITEM, ID, UNSEEN, SEEN
from freerec.utils import mkdirs, timemeter
from freeplot.utils import export_pickle, import_pickle


freerec.declare(version="0.4.3")


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--hidden-dim", type=int, default=512)
cfg.add_argument("--layers", type=int, default=2, help="L")
cfg.add_argument("--num-filters", type=int, default=500, help="top-K singular vectors")
cfg.add_argument("--dropout-rate", type=float, default=0.2)
cfg.add_argument("--upper", type=float, default=1., help="upper bound of beta")
cfg.add_argument("--lower", type=float, default=0., help="lower bound of beta")
cfg.add_argument("--weight", type=float, default=1, help="lambda")
cfg.add_argument("--alpha", type=float, default=15.)
cfg.set_defaults(
    description="StableGCN",
    root="../data",
    dataset='Gowalla_m1',
    epochs=200,
    batch_size=2048,
    optimizer='sgd',
    lr=0.1,
    weight_decay=2.e-4,
    eval_freq=5,
    seed=1
)
cfg.compile()

if cfg.upper < cfg.lower:
    raise ValueError(f"Check lower and upper bound of beta ...")


class BipartiteNorm(torch.nn.Module):

    def __init__(self, num_features: int, index: int) -> None:
        super().__init__()

        self.UserNorm = torch.nn.BatchNorm1d(num_features)
        self.ItemNorm = torch.nn.BatchNorm1d(num_features)
        self.index = index

    def forward(self, x: torch.Tensor):
        users, items = x[:self.index], x[self.index:]
        users, items = self.UserNorm(users), self.ItemNorm(items)
        return torch.cat((users, items), dim=0)


class StableGCN(RecSysArch):

    def __init__(
        self, fields: FieldModuleList, 
        graph: Data,
    ) -> None:
        super().__init__()

        self.fields = fields
        self.num_layers = cfg.layers
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]
        self.graph = graph

        self.dense = torch.nn.Sequential(
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.num_filters, cfg.hidden_dim, bias=False),
            BipartiteNorm(cfg.hidden_dim, self.User.count),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False),
            BipartiteNorm(cfg.hidden_dim, self.User.count),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim, bias=False),
        )
        self.conv = LGConv(normalize=False)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph
        T.ToSparseTensor()(self.__graph)
        self.__graph.adj_t = gcn_norm(
            self.__graph.adj_t, num_nodes=self.User.count + self.Item.count,
            add_self_loops=False
        )

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device:
            self.graph.to(device)
        return super().to(device, dtype, non_blocking)

    def save(self, data: Dict):
        path = os.path.join("filters", cfg.dataset, str(int(cfg.alpha)))
        mkdirs(path)
        file_ = os.path.join(path, "u_s_v.pickle")
        export_pickle(data, file_)

    def load(self, graph: HeteroData):
        path = os.path.join("filters", cfg.dataset, str(int(cfg.alpha)))
        file_ = os.path.join(path, "u_s_v.pickle")
        try:
            data = import_pickle(file_)
        except ImportError:
            data = self.preprocess(graph)
            self.save(data)
        
        U, vals, V = data['U'], data['vals'], data['V']
        vals = self.weight_filter(vals[:cfg.num_filters])
        U = U[:, :cfg.num_filters] * vals * math.sqrt(U.size(0))
        V = V[:, :cfg.num_filters] * vals * math.sqrt(V.size(0))
        self.register_buffer("U", U)
        self.register_buffer("V", V)

    def weight_filter(self, vals: torch.Tensor):
        return vals.div(math.sqrt(cfg.num_filters))

    def preprocess(self, graph: HeteroData):
        R = sp.lil_array(to_scipy_sparse_matrix(
            graph[graph.edge_types[0]].edge_index,
            num_nodes=max(self.User.count, self.Item.count)
        ))[:self.User.count, :self.Item.count] # N x M
        userDegs = R.sum(axis=1).squeeze() + cfg.alpha
        itemDegs = R.sum(axis=0).squeeze() + cfg.alpha
        userDegs = 1 / np.sqrt(userDegs)
        itemDegs = 1 / np.sqrt(itemDegs)
        userDegs[np.isinf(userDegs)] = 0.
        itemDegs[np.isinf(itemDegs)] = 0.
        R = (userDegs.reshape(-1, 1) * R * itemDegs).tocoo()
        rows = torch.from_numpy(R.row).long()
        cols = torch.from_numpy(R.col).long()
        vals = torch.from_numpy(R.data)
        indices = torch.stack((rows, cols), dim=0)
        R = torch.sparse_coo_tensor(
            indices, vals, size=R.shape
        )

        U, vals, V = torch.svd_lowrank(R, q=1000, niter=30)

        data = {'U': U.cpu(), 'vals': vals.cpu(), 'V': V.cpu()}
        return data

    def score(self, feats, users, items):
        userFeats, itemFeats = torch.split(feats, (self.User.count, self.Item.count))
        userFeats = userFeats[users] # B x 1 x D
        itemFeats = itemFeats[items] # B x n x D
        return torch.mul(userFeats, itemFeats).sum(-1)

    def forward(
        self, users: torch.Tensor,
        items: torch.Tensor
    ):
        userEmbs = self.U
        itemEmbs = self.V
        embds = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        embds = self.dense(embds)
        features = embds
        avgFeats = embds
        for _ in range(self.num_layers):
            features = self.conv(features, self.graph.adj_t) * cfg.weight / (cfg.weight + 1)
            avgFeats = avgFeats + features
        avgFeats = avgFeats / (cfg.weight + 1)

        scoress = self.score(embds, users, items)
        scores = self.score(avgFeats, users, items)
        return scoress, scores

    def recommend(self):
        userEmbs = self.U
        itemEmbs = self.V
        embds = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        embds = self.dense(embds)
        features = embds
        avgFeats = embds
        for _ in range(self.num_layers):
            features = self.conv(features, self.graph.adj_t) * cfg.weight / (cfg.weight + 1)
            avgFeats = avgFeats + features
        avgFeats = avgFeats / (cfg.weight + 1)
        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))
        return userFeats, itemFeats


class CoachForStableGCN(Coach):

    def prepare(self, T_max):
        self.T_cur = 0
        self.T_max = T_max

    def topdown(self):
        weight = cfg.upper - (self.T_cur / self.T_max) * (cfg.upper - cfg.lower)
        self.T_cur += 1
        return weight

    def train_per_epoch(self, epoch):
        count = 0
        for users, positives, negatives in self.dataloader:
            users = users.to(self.device)
            items = torch.cat((positives, negatives), dim=-1).to(self.device)

            weight = self.topdown()
            scoress, scores = self.model(users, items)
            positives, negatives = scores[:, 0], scores[:, 1]
            loss1 = self.criterion(positives, negatives)
            positives, negatives = scoress[:, 0], scoress[:, 1]
            loss2 = self.criterion(positives, negatives)
            loss = loss1 * (1 - weight) + loss2 * weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss1.item(), n=scores.size(0), mode="mean", prefix='train', pool=['LOSS1'])
            self.monitor(loss2.item(), n=scores.size(0), mode="mean", prefix='train', pool=['LOSS2'])
            self.monitor(loss.item(), n=scores.size(0), mode="mean", prefix='train', pool=['LOSS'])

            self.lr_scheduler.step()
            count += 1
        
    def evaluate(self, epoch, prefix: str = 'valid'):
        userFeats, itemFeats = self.model.recommend()
        for user, unseen, seen in self.dataloader:
            users = user.to(self.device).data
            seen = seen.to_csr().to(self.device).to_dense().bool()
            targets = unseen.to_csr().to(self.device).to_dense()
            users = userFeats[users].flatten(1) # B x D
            items = itemFeats.flatten(1) # N x D
            preds = users @ items.T # B x N
            preds[seen] = -1e10

            self.monitor(
                preds, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['NDCG', 'RECALL']
            )


def main():

    dataset = getattr(freerec.data.datasets.general, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = RandomIDs(
        field=User, datasize=dataset.train().datasize
    ).sharding_filter().gen_train_uniform_sampling_(
        dataset, num_negatives=1
    ).batch(cfg.batch_size).column_().tensor_()

    #validpipe
    validpipe = OrderedIDs(
        field=User
    ).sharding_filter().gen_valid_yielding_(
        dataset # return (user, unseen, seen)
    ).batch(128).column_().tensor_().field_(
        User.buffer(), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
    )

    testpipe = OrderedIDs(
        field=User
    ).sharding_filter().gen_test_yielding_(
        dataset # return (user, unseen, seen)
    ).batch(1024).column_().tensor_().field_(
        User.buffer(), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
    )

    tokenizer = FieldModuleList(dataset.fields.groupby(ID))
    model = StableGCN(
        tokenizer, dataset.train().to_graph((USER, ID), (ITEM, ID))
    )
    model.load(
        dataset.train().to_bigraph((USER, ID), (ITEM, ID))
    )

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    if cfg.optimizer == 'sgd':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * math.ceil(dataset.train().datasize / cfg.batch_size), eta_min=0.001)
    criterion = BPRLoss()

    coach = CoachForStableGCN(
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        fields=dataset.fields,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=cfg.device
    )
    coach.prepare(cfg.epochs * math.ceil(dataset.train().datasize / cfg.batch_size))
    coach.register_metric('loss1', func=lambda x: x, best_caster=min, fmt='.5f')
    coach.register_metric('loss2', func=lambda x: x, best_caster=min, fmt='.5f')
    coach.compile(cfg, monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'], which4best='ndcg@20')
    coach.fit()


if __name__ == "__main__":
    main()