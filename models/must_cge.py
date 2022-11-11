import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import MLP
from .gcn import GCN, GCNII
from .word_embedding import load_word_embeddings
import scipy.sparse as sp


def adj_to_edges(adj):
    # Adj sparse matrix to list of edges
    rows, cols = np.nonzero(adj)
    edges = list(zip(rows.tolist(), cols.tolist()))
    return edges


def edges_to_adj(edges, n):
    # List of edges to Adj sparse matrix
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype='float32')
    return adj


device = 'cuda' if torch.cuda.is_available() else 'cpu'
class MultiCEFocalLoss_branch(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss_branch, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1).cuda()
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target,compose_pred):
        pt = F.softmax(predict, dim=1)
        class_mask = F.one_hot(target, self.class_num)
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)]
        probs = (pt * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        probs_compose = (compose_pred * class_mask).sum(1).view(-1, 1)
        loss = -alpha * (torch.pow((1 - probs_compose), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1).cuda()
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target,obj_pred,attr_pred):
        pt = F.softmax(predict, dim=1)
        class_mask = F.one_hot(target, self.class_num)
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].view(-1,1)
        probs = (pt * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        probs_obj = (obj_pred * class_mask).sum(1).view(-1, 1)
        probs_attr = (attr_pred * class_mask).sum(1).view(-1, 1)

        loss = -alpha * (torch.pow((1 - probs_obj), self.gamma/2)) * (torch.pow((1-probs_attr),self.gamma/2)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class MUST_CGE(nn.Module):
    def __init__(self, dset, args):
        super(MUST_CGE, self).__init__()
        self.args = args
        self.dset = dset
        self.val_forward = self.val_forward_dotpr
        self.train_forward = self.train_forward_normal
        # Image Embedder
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        self.pairs = dset.pairs
        if self.args.train_only:
            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current]+self.num_attrs+self.num_objs)
            self.train_idx = torch.LongTensor(train_idx).to(device)
        self.args.fc_emb = self.args.fc_emb.split(',')
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)
        if args.nlayers:
            self.image_embedder = MLP(dset.feat_dim, args.emb_dim, num_layers= args.nlayers, dropout = self.args.dropout,
                norm = self.args.norm, layers = layers, relu = True)
        all_words = list(self.dset.attrs) + list(self.dset.objs)
        self.displacement = len(all_words)
        self.obj_to_idx = {word: idx for idx, word in enumerate(self.dset.objs)}
        self.attr_to_idx = {word: idx for idx, word in enumerate(self.dset.attrs)}

        if args.graph_init is not None:
            path = args.graph_init
            graph = torch.load(path)
            embeddings = graph['embeddings'].to(device)
            adj = graph['adj']
            self.embeddings = embeddings
        else:
            embeddings = self.init_embeddings(all_words).to(device)
            adj = self.adj_from_pairs()
            self.embeddings = embeddings

        hidden_layers = self.args.gr_emb
        if args.gcn_type == 'gcn':
            self.gcn = GCN(adj, self.embeddings.shape[1], args.emb_dim, hidden_layers)
        else:
            self.gcn = GCNII(adj, self.embeddings.shape[1], args.emb_dim, args.hidden_dim, args.gcn_nlayers, lamda = 0.5, alpha = 0.1, variant = False)


        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        layers2 = []
        self.args.attr_objs_fc_emb = self.args.attr_objs_fc_emb.split(',')
        for dim in self.args.attr_objs_fc_emb:
            dim = int(dim)
            layers2.append(dim)

        self.obj_img_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=False, num_layers=2,
                                  dropout=self.args.dropout,
                                  norm=self.args.norm, layers=layers2)
        self.attr_img_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=False, num_layers=2,
                                    dropout=self.args.dropout,
                                    norm=self.args.norm, layers=layers2)
        self.obj_pairs = self.create_obj_pairs().cuda()
        self.attr_pairs = self.create_attr_pairs().cuda()

        branch_emb_init = self.args.branch_emb_init
        if branch_emb_init == 'word2vec':
            embed_dim = 300
        else:
            embed_dim = 600
        self.attr_embedder = nn.Embedding(len(dset.attrs), embed_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), embed_dim)
        self.obj_projection = nn.Linear(embed_dim, args.emb_dim)
        self.attr_projection = nn.Linear(embed_dim,args.emb_dim)
        # init with word embeddings
        #ft+w2v
        #word2vec

        pretrained_weight = load_word_embeddings(branch_emb_init, dset.attrs)
        self.attr_embedder.weight.data.copy_(pretrained_weight)
        pretrained_weight = load_word_embeddings(branch_emb_init, dset.objs)
        self.obj_embedder.weight.data.copy_(pretrained_weight)
        self.focal = MultiCEFocalLoss(class_num=len(self.train_idx),gamma=2).cuda()
        self.focal_branch_attr = MultiCEFocalLoss_branch(class_num=self.num_attrs,gamma=1).cuda()
        self.focal_branch_objs = MultiCEFocalLoss_branch(class_num=self.num_objs,gamma=1).cuda()
        self._train_idx = self.train_idx-self.num_objs-self.num_attrs
        self.obj_pairs_train = self.obj_pairs[:,self._train_idx]
        self.attr_pairs_train = self.attr_pairs[:,self._train_idx]
    def create_obj_pairs(self):
        obj_matrix = torch.zeros(self.num_objs,self.num_pairs)
        for i in range(self.num_objs):
            for j in range(self.num_pairs):
                if self.dset.objs[i] == self.pairs[j][1]:
                    obj_matrix[i,j] = 1
        return obj_matrix
    def create_attr_pairs(self):
        obj_matrix = torch.zeros(self.num_attrs,self.num_pairs)
        for i in range(self.num_attrs):
            for j in range(self.num_pairs):
                if self.dset.attrs[i] == self.pairs[j][0]:
                    obj_matrix[i,j] = 1
        return obj_matrix
    def init_embeddings(self, all_words):

        def get_compositional_embeddings(embeddings, pairs):
            # Getting compositional embeddings from base embeddings
            composition_embeds = []
            for (attr, obj) in pairs:
                attr_embed = embeddings[self.attr_to_idx[attr]]
                obj_embed = embeddings[self.obj_to_idx[obj]+self.num_attrs]
                composed_embed = (attr_embed + obj_embed) / 2
                composition_embeds.append(composed_embed)
            composition_embeds = torch.stack(composition_embeds)
            print('Compositional Embeddings are ', composition_embeds.shape)
            return composition_embeds

        # init with word embeddings
        embeddings = load_word_embeddings(self.args.emb_init, all_words)

        composition_embeds = get_compositional_embeddings(embeddings, self.pairs)
        full_embeddings = torch.cat([embeddings, composition_embeds], dim=0)

        return full_embeddings


    def update_dict(self, wdict, row,col,data):
        wdict['row'].append(row)
        wdict['col'].append(col)
        wdict['data'].append(data)

    def compose(self, img):
        logits = []
        pred = []

        obj_img = F.normalize(self.obj_img_embedder(img),dim=1)
        objs_ = F.normalize(self.obj_projection(F.leaky_relu(self.obj_embedder(self.uniq_objs),inplace=True)),dim=1)

        attr_img = F.normalize(self.attr_img_embedder(img),dim=1)
        attr_ = F.normalize(self.attr_projection(F.leaky_relu(self.attr_embedder(self.uniq_attrs),inplace=True)),dim=1)
        logits.append(torch.matmul(obj_img,objs_.T))
        logits.append(torch.matmul(attr_img, attr_.T))
        pred.append(logits[0]@self.obj_pairs)
        pred.append(logits[1]@self.attr_pairs)
        return logits,pred

    def adj_from_pairs(self):

        def edges_from_pairs(pairs):
            weight_dict = {'data':[],'row':[],'col':[]}


            for i in range(self.displacement):
                self.update_dict(weight_dict,i,i,1.)

            for idx, (attr, obj) in enumerate(pairs):
                attr_idx, obj_idx = self.attr_to_idx[attr], self.obj_to_idx[obj] + self.num_attrs

                self.update_dict(weight_dict, attr_idx, obj_idx, 1.)
                self.update_dict(weight_dict, obj_idx, attr_idx, 1.)

                node_id = idx + self.displacement
                self.update_dict(weight_dict,node_id,node_id,1.)

                self.update_dict(weight_dict, node_id, attr_idx, 1.)
                self.update_dict(weight_dict, node_id, obj_idx, 1.)


                self.update_dict(weight_dict, attr_idx, node_id, 1.)
                self.update_dict(weight_dict, obj_idx, node_id, 1.)

            return weight_dict

        edges = edges_from_pairs(self.pairs)
        adj = sp.csr_matrix((edges['data'], (edges['row'], edges['col'])),
                            shape=(len(self.pairs)+self.displacement, len(self.pairs)+self.displacement))

        return adj

    def train_forward_normal(self, x,mode = 'warming_up'):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        if self.args.nlayers:
            img_feats = self.image_embedder(img)
        else:
            img_feats = (img)
        _train_idx = self.train_idx-self.num_objs-self.num_attrs
        current_embeddings = self.gcn(self.embeddings)
        logits, _  = self.compose(img)
        if self.args.train_only:
            pair_embed = current_embeddings[self.train_idx]
        else:
            pair_embed = current_embeddings[self.num_attrs+self.num_objs:self.num_attrs+self.num_objs+self.num_pairs,:]

        pair_embed = pair_embed.permute(1,0)
        pair_pred = torch.matmul(img_feats, pair_embed)

        if mode == 'warming_up':
            loss2 = F.cross_entropy(self.args.tem * logits[0], objs)
            loss3 = F.cross_entropy(self.args.tem * logits[1], attrs)
        else:
            attr_2_obj = F.softmax(logits[1],dim=1) @ self.attr_pairs @ self.obj_pairs.T
            obj_2_attr = F.softmax(logits[0],dim=1) @ self.obj_pairs @ self.attr_pairs.T
            loss2 = self.args.l2_weights*self.focal_branch_objs(self.args.tem*logits[0],objs,attr_2_obj.detach())
            loss3 = self.args.l2_weights*self.focal_branch_attr(self.args.tem*logits[1],attrs,obj_2_attr.detach())

        pred = []
        logits[0] = F.softmax(logits[0],dim=1)
        logits[1] = F.softmax(logits[1],dim=1)
        pred.append(logits[0]@self.obj_pairs)
        pred.append(logits[1]@self.attr_pairs)
        pred[0] = pred[0][:, self._train_idx]
        pred[1] = pred[1][:, self._train_idx]

        total_pred = pair_pred#+0.2*pred[0]+0.1*pred[1]
        loss4 = self.focal(total_pred,pairs,pred[0].detach(),pred[1].detach())

        return  loss2+loss3+loss4, None

    def val_forward_dotpr(self, x):
        img = x[0]

        if self.args.nlayers:
            img_feats = self.image_embedder(img)
        else:
            img_feats = (img)
        _, pred = self.compose(img)
        current_embedddings = self.gcn(self.embeddings)

        pair_embeds = current_embedddings[self.num_attrs+self.num_objs:self.num_attrs+self.num_objs+self.num_pairs,:].permute(1,0)

        score = torch.matmul(img_feats, pair_embeds)
        information_gain = torch.zeros_like(score)
        max_obj,_ = torch.max(pred[0],dim=1)
        max_attr,_ = torch.max(pred[1],dim=1)

        p = []
        for i in range(score.shape[0]):
            p.append(max_obj[i]/(max_obj[i]+max_attr[i]))
        for i in range(score.shape[0]):
            if max_obj[i]>=max_attr[i]:
                information_gain[i] = pred[0][i]*p[i]+pred[1][i]*(1-p[i])
            else:
                information_gain[i] = pred[1][i]*(1-p[i])+pred[0][i]*p[i]
        score = score + information_gain*self.args.objs_weights
        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:,self.dset.all_pair2idx[pair]]

        return None, scores

    def forward(self, x, mode='train'):
        if self.training:
            loss, pred = self.train_forward(x,mode)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred
