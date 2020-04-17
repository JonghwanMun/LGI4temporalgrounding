import pdb
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import building_networks as bn
from src.utils import net_utils

def basic_block(idim, odim, ksize=3):
    layers = []
    # 1st conv
    p = ksize // 2
    layers.append(nn.Conv1d(idim, odim, ksize, 1, p, bias=False))
    layers.append(nn.BatchNorm1d(odim))
    layers.append(nn.ReLU(inplace=True))
    # 2nd conv
    layers.append(nn.Conv1d(odim, odim, ksize, 1, p, bias=False))
    layers.append(nn.BatchNorm1d(odim))

    return nn.Sequential(*layers)


class ResBlock1D(nn.Module):
    def __init__(self, config, prefix=""):
        super(ResBlock1D, self).__init__() # Must call super __init__()
        name = prefix if prefix is "" else prefix+"_"

        # get configuration
        idim = config.get(name+"res_block_1d_idim", -1)
        odim = config.get(name+"res_block_1d_odim", -1)
        hdim = config.get(name+"res_block_1d_hdim", -1)
        ksize = config.get(name+"res_block_1d_ksize", 3)
        self.nblocks = config.get(name+"num_res_blocks", 1)
        self.do_downsample = config.get(name+"do_downsample", False)

        # set layers
        if self.do_downsample:
            self.downsample = nn.Sequential(
                nn.Conv1d(idim, odim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(odim),
            )
        self.blocks = nn.ModuleList()
        for i in range(self.nblocks):
            cur_block = basic_block(idim, odim, ksize)
            self.blocks.append(cur_block)
            if (i == 0) and self.do_downsample:
                idim = odim

    def forward(self, inp):
        """
        Args:
            inp: [B, idim, H, w]
        Returns:
            answer_label : [B, odim, H, w]
        """
        residual = inp
        for i in range(self.nblocks):
            out = self.blocks[i](residual)
            if (i == 0) and self.do_downsample:
                residual = self.downsample(residual)
            out += residual
            out = F.relu(out) # w/o is sometimes better
            residual = out

        return out


class AttentivePooling(nn.Module):
    def __init__(self, config, prefix=""):
        super(AttentivePooling, self).__init__()
        name = prefix if prefix is "" else prefix+"_"
        print("Attentive Poolig - ", name)

        self.att_n = config.get(name+"att_n", 1)
        self.feat_dim = config.get(name+"att_cand_dim", -1)
        self.att_hid_dim = config.get(name+"att_hdim", -1)
        self.use_embedding = config.get(name+"att_use_embedding", True)

        self.feat2att = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)
        self.to_alpha = nn.Linear(self.att_hid_dim, self.att_n, bias=False)
        if self.use_embedding:
            edim = config.get(name+"att_edim", 512)
            self.fc = nn.Linear(self.feat_dim, edim)

    def forward(self, feats, f_masks=None):
        """ Compute attention weights and attended feature (weighted sum)
        Args:
            feats: features where attention weights are computed; [B, A, D]
            f_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert f_masks is None or len(f_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)

        # embedding feature vectors
        attn_f = self.feat2att(feats)   # [B,A,hdim]

        # compute attention weights
        dot = torch.tanh(attn_f)        # [B,A,hdim]
        alpha = self.to_alpha(dot)      # [B,A,att_n]
        if f_masks is not None:
            alpha = alpha.masked_fill(f_masks.float().unsqueeze(2).eq(0), -1e9)
        attw =  F.softmax(alpha.transpose(1,2), dim=2) # [B,att_n,A]

        att_feats = attw @ feats # [B,att_n,D]
        if self.att_n == 1:
            att_feats = att_feats.squeeze(1)
            attw = attw.squeeze(1)
        if self.use_embedding: att_feats = self.fc(att_feats)

        return att_feats, attw

class Attention(nn.Module):
    def __init__(self, config, prefix=""):
        super(Attention, self).__init__()
        name = prefix if prefix is "" else prefix+"_"
        print("Attention - ", name)

        # parameters
        kdim = config.get(name+"att_key_dim", -1)
        cdim = config.get(name+"att_cand_dim", -1)
        att_hdim = config.get(name+"att_hdim", -1)
        drop_p = config.get(name+"att_drop_prob", 0.0)

        # layers
        self.key2att = nn.Linear(kdim, att_hdim)
        self.feat2att = nn.Linear(cdim, att_hdim)
        self.to_alpha = nn.Linear(att_hdim, 1)
        self.drop = nn.Dropout(drop_p)

    def forward(self, key, feats, feat_masks=None, return_weight=True):
        """ Compute attention weights and attended feature (weighted sum)
        Args:
            key: key vector to compute attention weights; [B, K]
            feats: features where attention weights are computed; [B, A, D]
            feat_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(key.size()) == 2, "{} != 2".format(len(key.size()))
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert feat_masks is None or len(feat_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)

        # compute attention weights
        logits = self.compute_att_logits(key, feats, feat_masks) # [B,A]
        weight = self.drop(F.softmax(logits, dim=1))             # [B,A]

        # compute weighted sum: bmm working on (B,1,A) * (B,A,D) -> (B,1,D)
        att_feats = torch.bmm(weight.unsqueeze(1), feats).squeeze(1) # B * D
        if return_weight:
            return att_feats, weight
        return att_feats

    def compute_att_logits(self, key, feats, feat_masks=None):
        """ Compute attention weights
        Args:
            key: key vector to compute attention weights; [B, K]
            feats: features where attention weights are computed; [B, A, D]
            feat_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(key.size()) == 2
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert feat_masks is None or len(feat_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)
        A = feats.size(1)

        # embedding key and feature vectors
        att_f = net_utils.apply_on_sequence(self.feat2att, feats)   # B * A * att_hdim
        att_k = self.key2att(key)                                   # B * att_hdim
        att_k = att_k.unsqueeze(1).expand_as(att_f)                 # B * A * att_hdim

        # compute attention weights
        dot = torch.tanh(att_f + att_k)                             # B * A * att_hdim
        alpha = net_utils.apply_on_sequence(self.to_alpha, dot)     # B * A * 1
        alpha = alpha.view(-1, A)                                   # B * A
        if feat_masks is not None:
            alpha = alpha.masked_fill(feat_masks.float().eq(0), -1e9)

        return alpha


class VideoEmbeddingWithPosition(nn.Module):
    def __init__(self, config, prefix=""):
        super(VideoEmbeddingWithPosition, self).__init__() # Must call super __init__()
        name = prefix if prefix is "" else prefix+"_"

        # get configuration
        v_idim = config.get(name+"vemb_idim")
        v_odim = config.get(name+"vemb_odim")
        self.use_position = config.get(name+"use_position", True)

        # define layers --- segment embedding and position embedding
        self.vid_emb_fn = nn.Sequential(*[
            nn.Linear(v_idim, v_odim),
            nn.ReLU(),
            nn.Dropout(0.5)
        ])
        if self.use_position:
            p_idim = config.get(name+"pemb_idim", -1)
            p_odim = config.get(name+"pemb_odim", -1)
            self.pos_emb_fn = nn.Sequential(*[
                nn.Embedding(p_idim, p_odim),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])

    def forward(self, seg_feats, seg_masks):
        """ encode video and return logits over proposals
        Args:
            seg_feats: segment-level features of video from 3D CNN; [B,L,v_idim]
            mask: mask for effective segments; [B,L]
        Returns:
            seg_emb: embedded segment-level feature (with position embedding); [B,L,v_odim]
        """

        # embedding segment-level features
        seg_emb = self.vid_emb_fn(seg_feats) * seg_masks.float().unsqueeze(2)

        if self.use_position:
            # use absolute position embedding
            pos = torch.arange(0, seg_masks.size(1)).type_as(seg_masks).unsqueeze(0).long()
            pos_emb = self.pos_emb_fn(pos)
            B, nseg, pdim = pos_emb.size()
            pos_feats = (pos_emb.expand(B, nseg, pdim) * seg_masks.unsqueeze(2).float())
            seg_emb += pos_feats

        return seg_emb


class QuerySequenceEncoder(nn.Module):
    """ RNN-based encoder network for sequence data (1D data, e.g., sentence)
    """
    def __init__(self, config, prefix=""):
        super(QuerySequenceEncoder, self).__init__()
        name = prefix if prefix is "" else prefix+"_"
        print("QuerySequenceEncoder - ", name)

        # define layers --- word embedding and RNN
        emb_idim =  config.get(name+"emb_idim", 512)
        emb_odim =  config.get(name+"emb_odim", 512)
        self.embedding = nn.Embedding(emb_idim, emb_odim)
        self.rnn = bn.get_rnn(config, prefix) # == LSTM

    def forward(self, onehot, mask, out_type="all_hidden"):
        """ encode query sequence using RNN and return logits over proposals
        Args:
            onehot: onehot vectors of query; [B, vocab_size]
            mask: mask for query; [B,L]
            out_type: output type [word-level | sentenve-level | both]
        Returns:
            w_feats: word-level features; [B,L,2*h]
            s_feats: sentence-level feature; [B,2*h]
        """
        # embedding onehot data
        wemb = self.embedding(onehot) # [B,L,emb_odim]

        # encoding onehot data.
        max_len = onehot.size(1) # == L
        length = mask.sum(1) # [B,]
        pack_wemb = nn.utils.rnn.pack_padded_sequence(
                wemb, length, batch_first=True, enforce_sorted=False)
        w_feats, _ = self.rnn(pack_wemb)
        w_feats, max_ = nn.utils.rnn.pad_packed_sequence(
                w_feats, batch_first=True, total_length=max_len)
        w_feats = w_feats.contiguous() # [B,L,2*h]

        if out_type == "word-level":
            return w_feats
        else:
            B, L, H = w_feats.size()
            idx = (length-1).long() # 0-indexed
            idx = idx.view(B, 1, 1).expand(B, 1, H//2)
            fLSTM = w_feats[:,:,:H//2].gather(1, idx).view(B, H//2)
            bLSTM = w_feats[:,0,H//2:].view(B,H//2)
            s_feats = torch.cat([fLSTM, bLSTM], dim=1)
            if out_type == "both":
                return w_feats, s_feats
            else:
                return s_feats

    def load_glove(self, wtoi, glove_path):
        """  Load pre-trained parameters of glove """
        # load glove word vector
        glove = torch.load(glove_path)
        g_words = glove["words"]
        g_wtoi = glove["wtoi"]
        g_vec = glove["vectors"] # [300, 40000]

        print("lookup table: ", self.embedding.weight.size())
        print("glove vector: ", g_vec.size())
        # note that the size of self.embedding.weight is [vocab_size, 300]
        cnt = 0
        for w in wtoi.keys():
            if w in g_words:
                self.embedding.weight.data[wtoi[w],:] = g_vec[:,g_wtoi[w]]
            else:
                #print("{} not in glove".format(w))
                self.embedding.weight.data[wtoi[w],:] = 0.0
                cnt = cnt + 1
        print("The number of non-existence words {}/{}".format(cnt, len(wtoi)))
        self.embedding.weight.data = net_utils.to_contiguous(self.embedding.weight.data)


class SequentialQueryAttention(nn.Module):
    def __init__(self, config):
        super(SequentialQueryAttention, self).__init__()

        self.nse = config.get("num_semantic_entity", -1)
        self.qdim = config.get("sqan_qdim", -1) # 512
        self.global_emb_fn = nn.ModuleList( # W_q^(n) in Eq. (4)
                [nn.Linear(self.qdim, self.qdim) for i in range(self.nse)])
        self.guide_emb_fn = nn.Sequential(*[
            nn.Linear(2*self.qdim, self.qdim), # W_g in Eq. (4)
            nn.ReLU()
        ])
        self.att_fn = Attention(config, "sqan")

    def forward(self, q_feats, w_feats, w_mask=None):
        """ extract N (=nse) semantic entity features from query
        Args:
            q_feats: sentence-level feature; [B,qdim]
            w_feats: word-level features; [B,L,qdim]
            w_mask: mask for effective words; [B,L]
        Returns:
            se_feats: semantic entity features; [B,N,qdim]
            se_attw: attention weight over words; [B,N,L]
        """

        B = w_feats.size(0)
        prev_se = w_feats.new_zeros(B, self.qdim)
        se_feats, se_attw = [], []
        # compute semantic entity features sequentially
        for n in range(self.nse):
            # perform Eq. (4)
            q_n = self.global_emb_fn[n](q_feats) # [B,qdim] -> [B,qdim]
            g_n = self.guide_emb_fn(torch.cat([q_n, prev_se], dim=1)) # [B,2*qdim] -> [B,qdim]
            # perform Eq. (5), (6), (7)
            att_f, att_w = self.att_fn(g_n, w_feats, w_mask)

            prev_se = att_f
            se_feats.append(att_f)
            se_attw.append(att_w)

        return torch.stack(se_feats, dim=1), torch.stack(se_attw, dim=1)


class HadamardProduct(nn.Module):
    def __init__(self, config, prefix=""):
        super(HadamardProduct, self).__init__() # Must call super __init__()
        name = prefix if prefix is "" else prefix+"_"
        print("Hadamard Product - ", name)

        idim_1 = config.get(name+"hp_idim_1", -1)
        idim_2 = config.get(name+"hp_idim_2", -1)
        hdim = config.get(name+"hp_hdim", -1) # 512

        self.fc_1 = nn.Linear(idim_1, hdim)
        self.fc_2 = nn.Linear(idim_2, hdim)
        self.fc_3 = nn.Linear(hdim, hdim)

    def forward(self, inp):
        """
        Args:
            inp1: [B,idim_1] or [B,L,idim_1]
            inp2: [B,idim_2] or [B,L,idim_2]
        """
        x1, x2 = inp[0], inp[1]
        return torch.relu(self.fc_3(torch.relu(self.fc_1(x1)) * torch.relu(self.fc_2(x2))))


class NonLocalBlock(nn.Module):
    def __init__(self, config, prefix=""):
        super(NonLocalBlock, self).__init__()
        name = prefix if prefix is "" else prefix+"_"
        print("Non-Local Block - ", name)

        # dims
        self.idim = config.get(name+"nl_idim", -1)
        self.odim = config.get(name+"nl_odim", -1)
        self.nheads = config.get(name+"nl_nheads", -1)

        # options
        self.use_bias = config.get(name+"nl_use_bias", True)
        self.use_local_mask = config.get(name+"nl_use_local_mask", False)
        if self.use_local_mask:
            self.ksize = config.get(name+"nl_mask_ksize", 15)
            self.dilation= config.get(name+"nl_mask_dilation", 1)
            self.local_mask = None

        # layers
        self.c_lin = nn.Linear(self.idim, self.odim*2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(config.get(name+"nl_drop_prob", 0.0))

    def forward(self, m_feats, mask):
        """
        Inputs:
            m_feats: segment-level multimodal feature     [B,nseg,*]
            mask: mask                              [B,nseg]
        Outputs:
            updated_m: updated multimodal  feature  [B,nseg,*]
        """

        mask = mask.float()
        B, nseg = mask.size()

        # key, query, value
        m_k = self.v_lin(self.drop(m_feats)) # [B,num_seg,*]
        m_trans = self.c_lin(self.drop(m_feats))  # [B,nseg,2*]
        m_q, m_v = torch.split(m_trans, m_trans.size(2) // 2, dim=2)

        new_mq = m_q
        new_mk = m_k

        # applying multi-head attention
        w_list = []
        mk_set = torch.split(new_mk, new_mk.size(2) // self.nheads, dim=2)
        mq_set = torch.split(new_mq, new_mq.size(2) // self.nheads, dim=2)
        mv_set = torch.split(m_v, m_v.size(2) // self.nheads, dim=2)
        for i in range(self.nheads):
            mk_slice, mq_slice, mv_slice = mk_set[i], mq_set[i], mv_set[i] # [B, nseg, *]

            # compute relation matrix; [B,nseg,nseg]
            m2m = mk_slice @ mq_slice.transpose(1,2) / ((self.odim // self.nheads) ** 0.5)
            if self.use_local_mask:
                local_mask = mask.new_tensor(self._get_mask(nseg, self.ksize, self.dilation)) # [nseg,nseg]
                m2m = m2m.masked_fill(local_mask.unsqueeze(0).eq(0), -1e9) # [B,nseg,nseg]
            m2m = m2m.masked_fill(mask.unsqueeze(1).eq(0), -1e9) # [B,nseg,nseg]
            m2m_w = F.softmax(m2m, dim=2) # [B,nseg,nseg]
            w_list.append(m2m_w)

            # compute relation vector for each segment
            r = m2m_w @ mv_slice if (i==0) else torch.cat((r, m2m_w @ mv_slice), dim=2)

        updated_m = self.drop(m_feats + r)
        return updated_m, torch.stack(w_list, dim=1)

    def _get_mask(self, N, ksize, d):
        if self.local_mask is not None: return self.local_mask
        self.local_mask = np.eye(N)
        K = ksize // 2
        for i in range(1, K+1):
            self.local_mask += np.eye(N, k=d+(i-1)*d)
            self.local_mask += np.eye(N, k=-(d+(i-1)*d))
        return self.local_mask # [N,N]

class LocalGlobalVideoTextInteractions(nn.Module):
    def __init__(self, config, prefix=""):
        super(LocalGlobalVideoTextInteractions, self).__init__()
        name = prefix if prefix is "" else prefix+"_"
        print("Local-Global Video-Text Interactions module - ", name)

        self.nse = config.get("num_semantic_entity", -1)

        # Multimodal fusion layer
        self.mm_fusion_method = config.get("lgi_fusion_method", "mul")
        if self.mm_fusion_method == "mul":
            self.fusion_fn = self._make_modulelist(
                    HadamardProduct(config, "lgi"), self.nse)
        elif self.mm_fusion_method == "concat":
            self.lin_fn = self._make_modulelist(nn.Linear(1024, 512), self.nse)

        # Local interaction layer
        self.l_type = config.get("lgi_local_type", "res_block")
        if self.l_type == "res_block":
            self.local_fn = self._make_modulelist(ResBlock1D(config, "lgi_local"), self.nse)
        elif self.l_type == "masked_nl":
            self.n_local_mnl = config.get("lgi_local_num_nl_block", -1)
            nth_local_fn = self._make_modulelist(
                    NonLocalBlock(config, "lgi_local"), self.n_local_mnl)
            self.local_fn = self._make_modulelist(nth_local_fn, self.nse)

        # Global interaction layer
        self.g_type = config.get("lgi_global_type", "nl")
        self.satt_fn = AttentivePooling(config, "lgi_global_satt")
        if self.g_type == "nl":
            self.n_global_nl = config.get("lgi_global_num_nl_block", -1)
            self.global_fn = self._make_modulelist(
                    NonLocalBlock(config, "lgi_global"), self.n_global_nl)

    def forward(self, seg_feats, seg_masks, se_feats):
        """ Perform local-global video-text interactions
        1) modality fusion, 2) local context modeling, and 3) global context modeling
        Args:
            seg_feats: segment-level features; [B,L,D]
            seg_masks: masks for effective segments in video; [B,L_v]
            se_feats: semantic entity features; [B,N,D]
        Returns:
            sa_feats: semantic-aware segment features; [B,L_v,D]
        """

        if self.nse == 1:
            se_feats = se_feats.unsqueeze(1)
        assert self.nse == se_feats.size(1)
        B, nseg, _ = seg_feats.size()

        m_feats = self._segment_level_modality_fusion(seg_feats, se_feats)
        ss_feats = self._local_context_modeling(m_feats, seg_masks)
        sa_feats, sattw = self._global_context_modeling(ss_feats, se_feats, seg_masks)

        return sa_feats, sattw

    def _make_modulelist(self, net, n):
        assert n > 0
        new_net_list = nn.ModuleList()
        new_net_list.append(net)
        if n > 1:
            for i in range(n-1):
                new_net_list.append(copy.deepcopy(net))
        return new_net_list

    def _segment_level_modality_fusion(self, s_feats, se_feats):
        B, nseg, _ = s_feats.size()
        # fuse segment-level feature with individual semantic entitiey features
        m_feats = []
        for n in range(self.nse):
            q4s_feat = se_feats[:,n,:].unsqueeze(1).expand(B, nseg, -1)
            if self.mm_fusion_method == "concat":
                fused_feat = torch.cat([s_feats, q4s_feat], dim=2)
                fused_feat = torch.relu(self.lin_fn[n](fused_feat))
            elif self.mm_fusion_method == "add":
                fused_feat = s_feats + q4s_feat
            elif self.mm_fusion_method == "mul":
                fused_feat = self.fusion_fn[n]([s_feats, q4s_feat])
            else:
                raise NotImplementedError()
            m_feats.append(fused_feat)

        return m_feats # N*[B*D]

    def _local_context_modeling(self, m_feats, masks):
        ss_feats = []

        for n in range(self.nse):
            if self.l_type == "res_block":
                l_feats = self.local_fn[n](m_feats[n].transpose(1,2)).transpose(1,2) # [B,nseg,*]
            elif self.l_type == "masked_nl":
                l_feats = m_feats[n]
                for s in range(self.n_local_mnl):
                    l_feats, _ = self.local_fn[n][s](l_feats, masks)
            else:
                l_feats = feats
            ss_feats.append(l_feats)

        return ss_feats # N*[B,D]

    def _global_context_modeling(self, ss_feats, se_feats, seg_masks):
        ss_feats = torch.stack(ss_feats, dim=1) # N*[B,nseg,D] -> [B,N,nseg,D]

        # aggregating semantics-specific features
        _, sattw = self.satt_fn(se_feats)
        # [B,N,1,1] * [B,N,nseg,D] = [B,N,nseg,D]
        a_feats = sattw.unsqueeze(2).unsqueeze(2) * ss_feats
        a_feats = a_feats.sum(dim=1) # [B,nseg,D]

        # capturing contextual and temporal relations between semantic entities
        if self.g_type == "nl":
            sa_feats = a_feats
            for s in range(self.n_global_nl):
                sa_feats, _ = self.global_fn[s](sa_feats, seg_masks)
        else:
            sa_feats = a_feats

        return sa_feats, sattw


class AttentionLocRegressor(nn.Module):
    def __init__(self, config, prefix=""):
        super(AttentionLocRegressor, self).__init__()
        name = prefix if prefix is "" else prefix+"_"
        print("Attention-based Location Regressor - ", name)

        self.tatt = AttentivePooling(config, "grounding")

        # Regression layer
        idim = config.get("grounding_idim", -1)
        gdim = config.get("grounding_hdim", 512)
        nn_list = [ nn.Linear(idim, gdim), nn.ReLU(), nn.Linear(gdim, 2)]
        if config["dataset"] == "charades":
            nn_list.append(nn.ReLU())
        else:
            nn_list.append(nn.Sigmoid())
        self.MLP_reg = nn.Sequential(*nn_list)

    def forward(self, semantic_aware_seg_feats, masks):
        # perform Eq. (13) and (14)
        summarized_vfeat, att_w = self.tatt(semantic_aware_seg_feats, masks)
        # perform Eq. (15)
        loc = self.MLP_reg(summarized_vfeat) # loc = [t^s, t^e]
        return loc, att_w


""" Criterions """
class MultipleCriterions(nn.Module):
    """ Container for multiple criterions.
    Since pytorch does not support ModuleDict(), we use ModuleList() to
    maintain multiple criterions.
    """
    def __init__(self, names=None, modules=None):
        super(MultipleCriterions, self).__init__()
        if names is not None:
            assert len(names) == len(modules)
        self.names = names if names is not None else []
        self.crits = nn.ModuleList(modules) if modules is not None else nn.ModuleList()
        self.name2crit = {}
        if names is not None:
            self.name2crit = {name:self.crits[i]for i,name in enumerate(names)}

    def forward(self, crit_inp, gts):
        self.loss = {}
        self.loss["total_loss"] = 0
        for name,crit in self.get_items():
            self.loss[name] = crit(crit_inp, gts)
            self.loss["total_loss"] += self.loss[name]
        return self.loss

    def add(self, name, crit):
        self.names.append(name)
        self.crits.append(crit)
        self.name2crit[name] = self.crits[-1]

    def get_items(self):
        return iter(zip(self.names, self.crits))

    def get_names(self):
        return self.names

    def get_crit_by_name(self, name):
        return self.name2crit[name]

class TGRegressionCriterion(nn.Module):
    """
    Loss function to compute weighted Binary Cross-Entropy loss
    for temporal grounding given language query
    """
    def __init__(self, cfg, prefix=""):
        super(TGRegressionCriterion, self).__init__()
        self.name = prefix if prefix is "" else prefix+"_"

        self.regloss1 = nn.SmoothL1Loss()
        self.regloss2 = nn.SmoothL1Loss()

    def forward(self, net_outs, gts):
        """ loss function to compute weighted Binary Cross-Entropy loss
        Args:
            net_outs: dictionary of network outputs
                - loc: location; [B,2] - start/end
            gts: dictionary of ground-truth
                - labels: grounding labels; [B,L_v,K], float tensor
        Returns:
            loss: loss value; [1], float tensor
        """
        loc  = net_outs[self.name + "loc"]    # [B,2]
        s_gt = gts[self.name + "start_pos"]     # [B]
        e_gt = gts[self.name + "end_pos"]       # [B]

        total_loss = self.regloss1(loc[:,0], s_gt) + self.regloss2(loc[:,1], e_gt)

        return total_loss

class DQALoss(nn.Module):
    def __init__(self, config, prefix=""):
        super(DQALoss, self).__init__()
        self.name = prefix if prefix is "" else prefix+"_"
        print("Distinct Query Attention Loss - ", self.name)

        self.w = config.get("dqa_weight", 1.0)
        self.r = config.get("dqa_lambda", 0.2)

    def forward(self, net_outs, gts):
        """ loss function to diversify attention weights
        Args:
            net_outs: dictionary of network outputs
            gts: dictionary of ground-truth
        Returns:
            loss: loss value; [1], float tensor
        """
        attw = net_outs[self.name+"dqa_attw"] # [B,num_att,N]
        NA = attw.size(1)

        attw_T = torch.transpose(attw, 1, 2).contiguous()

        I = torch.eye(NA).unsqueeze(0).type_as(attw) * self.r
        #pdb.set_trace()
        P = torch.norm(torch.bmm(attw, attw_T) - I, p="fro", dim=[1,2], keepdim=True)
        #P = torch.norm(torch.bmm(attw, attw_T) - I, p=2, dim=[1,2], keepdim=True)
        #P = torch.bmm(attw, attw_T) - I
        #P = torch.norm(P.cpu(), p="fro", dim=[1,2], keepdim=True).cuda()

        if torch.isnan(P).sum() > 0:
            print("attw: ", attw)
            pdb.set_trace()

        da_loss = self.w * (P**2).mean()

        return da_loss

class TAGLoss(nn.Module):
    def __init__(self, cfg, prefix=""):
        super(TAGLoss, self).__init__()
        self.name = prefix if prefix is "" else prefix+"_"
        print("Temporal Attention Guidance Loss - ", self.name)

        self.w = cfg.get("tag_weight", 1.0)

    def forward(self, net_outs, gts):
        mask = gts[self.name+"tag_att_masks"] # [B,num_segment]
        w = net_outs[self.name+"tag_attw"]  # [B,num_segment]

        ac_loss = (-mask*torch.log(w+1e-8)).sum(1) / mask.sum(1)
        ac_loss = (self.w * ac_loss.mean(0))

        return ac_loss
