import os
from collections import OrderedDict

from src.model import building_blocks as bb
from src.model.abstract_network import AbstractNetwork
from src.utils import io_utils, net_utils, vis_utils

class LGI(AbstractNetwork):
    def __init__(self, config, logger=None, verbose=True):
        """ Initialize baseline network for Temporal Language Grounding
        """
        super(LGI, self).__init__(config=config, logger=logger)

        self._build_network()
        self._build_evaluator()

        # create counters and initialize status
        self._create_counters()
        self.reset_status(init_reset=True)

    def _build_network(self):
        """ build network that consists of following four components
        1. encoders - query_enc & video enc
        2. sequential query attentio network (sqan)
        3. local-global video-text interactions layer (vti_fn)
        4. temporal attentive localization by regression (talr)
        """
        mconfig = self.config["model"]

        # build video & query encoder
        self.query_enc = bb.QuerySequenceEncoder(mconfig, "query_enc")
        self.video_enc = bb.VideoEmbeddingWithPosition(mconfig, "video_enc")

        # build sequential query attention network (sqan)
        self.nse = mconfig.get("num_semantic_entity", -1) # number of semantic phrases
        if self.nse > 1:
            self.sqan = bb.SequentialQueryAttention(mconfig)

        # build local-global video-text interactions network (vti_fn)
        self.vti_fn = bb.LocalGlobalVideoTextInteractions(mconfig)

        # build grounding fn
        self.ta_reg_fn = bb.AttentionLocRegressor(mconfig)

        # build criterion
        self.use_tag_loss = mconfig.get("use_temporal_attention_guidance_loss", True)
        self.use_dqa_loss = mconfig.get("use_distinct_query_attention_loss", True)
        self.criterion = bb.MultipleCriterions(
            ["grounding"],
            [bb.TGRegressionCriterion(mconfig, prefix="grounding")]
        )
        if self.use_tag_loss:
            self.criterion.add("tag", bb.TAGLoss(mconfig))
        if self.use_dqa_loss:
            self.criterion.add("dqa", bb.DQALoss(mconfig))

        # set model list
        self.model_list = ["video_enc", "query_enc",
                           "vti_fn", "ta_reg_fn", "criterion"]
        self.models_to_update = ["video_enc", "query_enc",
                                 "vti_fn", "ta_reg_fn", "criterion"]
        if self.nse > 1:
            self.model_list.append("sqan")
            self.models_to_update.append("sqan")

        self.log("===> We train [{}]".format("|".join(self.models_to_update)))

    def forward(self, net_inps):
        return self._infer(net_inps, "forward")

    def visualize(self, vis_inps, vis_gt, prefix):
        vis_data = self._infer(vis_inps, "visualize", vis_gt)
        vis_utils.visualize_LGI(self.config, vis_data, self.itow, prefix)

    def extract_output(self, vis_inps, vis_gt, save_dir):
        vis_data = self._infer(vis_inps, "save_output", vis_gt)

        qids = vis_data["qids"]
        preds = net_utils.loc2mask(loc, seg_masks)
        for i,qid in enumerate(qids):
            out = dict()
            for k in vis_data.keys():
                out[k] = vis_data[k][i]
            # save output
            save_path = os.path.join(save_dir, "{}.pkl".format(qid))
            io_utils.check_and_create_dir(save_dir)
            io_utils.write_pkl(save_path, out)

    def _infer(self, net_inps, mode="forward", gts=None):
        # fetch inputs
        word_labels = net_inps["query_labels"] # [B,L] (nword == L)
        word_masks = net_inps["query_masks"] # [B,L]
        c3d_feats = net_inps["video_feats"]  # [B,T,d_v]
        seg_masks = net_inps["video_masks"].squeeze(2) # [B,T]
        B, nseg, _ = c3d_feats.size() # nseg == T

        # forward encoders
        # get word-level, sentence-level and segment-level features
        word_feats, sen_feats = self.query_enc(word_labels, word_masks, "both") # [B,L,*]
        seg_feats = self.video_enc(c3d_feats, seg_masks) # [B,nseg,*]

        # get semantic phrase features:
        # se_feats: semantic phrase features [B,nse,*];
        #           ([e^1,...,e^n]) in Eq. (7)
        # se_attw: attention weights for semantic phrase [B,nse,nword];
        #           ([a^1,...,a^n]) in Eq. (6)
        if self.nse > 1:
            se_feats, se_attw = self.sqan(sen_feats, word_feats, word_masks)
        else: se_attw = None

        # Local-global video-text interactions
        # sa_feats: semantics-aware segment features [B,nseg,d]; R in Eq. (12)
        # s_attw: aggregating weights [B,nse]
        if self.nse > 1:
            q_feats = se_feats
        else:
            q_feats = sen_feats
        sa_feats, s_attw = self.vti_fn(seg_feats, seg_masks, q_feats)

        # Temporal attentive localization by regression
        # loc: prediction of time span (t^s, t^e)
        # t_attw: temporal attention weights (o)
        loc, t_attw = self.ta_reg_fn(sa_feats, seg_masks)

        if mode == "forward":
            outs = OrderedDict()
            outs["grounding_loc"] = loc
            if self.use_tag_loss:
                outs["tag_attw"] = t_attw
            if self.use_dqa_loss:
                outs["dqa_attw"] = se_attw
        else:
            outs = dict()
            outs["vids"] = gts["vids"]
            outs["qids"] = gts["qids"]
            outs["query_labels"] = net_utils.to_data(net_inps["query_labels"])
            outs["grounding_gt"] = net_utils.to_data(gts["grounding_att_masks"])
            outs["grounding_pred"] = net_utils.loc2mask(loc, seg_masks)
            outs["nfeats"] = gts["nfeats"]
            if self.nse > 1:
                outs["se_attw"] = net_utils.to_data(se_attw)
            else:
                outs["se_attw"] = net_utils.to_data(t_attw.new_zeros(t_attw.size(0),2,4))
            outs["t_attw"] = net_utils.to_data(t_attw.unsqueeze(1))
            if s_attw is None:
                outs["s_attw"] = net_utils.to_data(t_attw.new_zeros(t_attw.size(0),2,4))
            else:
                outs["s_attw"] = net_utils.to_data(s_attw)

            if mode == "save_output":
                outs["duration"] = gts["duration"]
                outs["timestamps"] = gts["timestamps"]
                outs["grounding_pred_loc"] = net_utils.to_data(loc)

        return outs

    def prepare_batch(self, batch):
        self.gt_list = ["vids", "qids", "timestamps", "duration",
                   "grounding_start_pos", "grounding_end_pos",
                   "grounding_att_masks", "nfeats"]
        self.both_list = ["grounding_att_masks"]

        net_inps, gts = {}, {}
        for k in batch.keys():
            item = batch[k].to(self.device) \
                if net_utils.istensor(batch[k]) else batch[k]

            if k in self.gt_list: gts[k] = item
            else: net_inps[k] = item

            if k in self.both_list: net_inps[k] = item

        if self.use_tag_loss:
            gts["tag_att_masks"] = gts["grounding_att_masks"]
        return net_inps, gts

    """ methods for status & counters """
    def reset_status(self, init_reset=False):
        """ Reset (initialize) metric scores or losses (status).
        """
        super(LGI, self).reset_status(init_reset=init_reset)

        # initialize prediction maintainer for each epoch
        self.results = {"predictions": [], "gts": [],
                        "durations": [], "vids": [], "qids": []}

    def compute_status(self, net_outs, gts, mode="Train"):

        # fetch data
        loc = net_outs["grounding_loc"].detach()
        B = loc.size(0)
        gt_ts = gts["timestamps"]
        vid_d = gts["duration"]

        # prepare results for evaluation
        for ii in range(B):
            pred = [[float(loc[ii,0])*vid_d[ii], float(loc[ii,1])*vid_d[ii]]]
            self.results["predictions"].append(pred)
            self.results["gts"].append(gt_ts[ii])
            self.results["durations"].append(vid_d[ii])
            self.results["vids"].append(gts["vids"][ii])
            self.results["qids"].append(gts["qids"][ii])

    def save_results(self, prefix, mode="Train"):
        # save predictions
        save_dir = os.path.join(self.config["misc"]["result_dir"], "predictions", mode)
        save_to = os.path.join(save_dir, prefix+".json")
        io_utils.check_and_create_dir(save_dir)
        io_utils.write_json(save_to, self.results)

        # compute performances
        nb = float(len(self.results["gts"]))
        self.evaluator.set_duration(self.results["durations"])
        rank1, rank5, miou = self.evaluator.eval(
                self.results["predictions"], self.results["gts"])

        for k,v  in rank1.items():
            self.counters[k].add(v/nb, 1)
        self.counters["mIoU"].add(miou/nb, 1)

    def renew_best_score(self):
        cur_score = self._get_score()
        if (self.best_score is None) or (cur_score > self.best_score):
            self.best_score = cur_score
            self.log("Iteration {}: New best score {:4f}".format(
                    self.it, self.best_score))
            return True
        self.log("Iteration {}: Current score {:4f}".format(self.it, cur_score))
        self.log("Iteration {}: Current best score {:4f}".format(
                self.it, self.best_score))
        return False

    """ methods for updating configuration """
    def bring_dataset_info(self, dset):
        super(LGI, self).bring_dataset_info(dset)

        # Query Encoder
        self.wtoi = dset.wtoi
        self.itow = dset.itow
        glove_path = self.config["model"].get("glove_path", "")
        if len(glove_path) > 0:
            self.query_enc.load_glove(self.wtoi, glove_path)

    def model_specific_config_update(self, config):
        mconfig = config["model"]
        ### Video Encoder
        vdim = mconfig["video_enc_vemb_odim"]
        ### Query Encoder
        mconfig["query_enc_rnn_idim"] = mconfig["query_enc_emb_odim"]
        qdim = 2 * mconfig["query_enc_rnn_hdim"]

        ### Sequential Query Attention Network (SQAN)
        mconfig["sqan_qdim"] = qdim
        mconfig["sqan_att_cand_dim"] = qdim
        mconfig["sqan_att_key_dim"] = qdim

        ### Local-Global Video-Text Interactions
        # Segment-level Modality Fusion
        self.lgi_fusion_method = mconfig.get("lgi_fusion_method", "concat")
        mdim = vdim # dim of fused multimodal feature
        if self.lgi_fusion_method == "mul":
            mconfig["lgi_hp_idim_1"] = vdim
            mconfig["lgi_hp_idim_2"] = qdim
            mconfig["lgi_hp_hdim"] = vdim
        # Local Context Modeling
        l_type = mconfig.get("lgi_local_type", "res_block")
        if l_type == "res_block":
            mconfig["lgi_local_res_block_1d_idim"] = mdim
            mconfig["lgi_local_res_block_1d_odim"] = mdim
            l_odim = mconfig["lgi_local_res_block_1d_odim"]
        elif l_type == "masked_nl":
            mconfig["lgi_local_nl_odim"] = mdim
            l_odim = mconfig["lgi_local_nl_odim"]
        else:
            l_odim = mdim

        # Global Context Modeling
        g_type = mconfig.get("lgi_global_type", "nl")
        mconfig["lgi_global_satt_att_cand_dim"] = l_odim
        mconfig["lgi_global_nl_idim"] = l_odim

        ### Temporal Attention based Regression
        mconfig["grounding_att_key_dim"] = qdim
        mconfig["grounding_att_cand_dim"] = qdim
        if mconfig.get("lgi_global_satt_att_use_embedding", False):
            mconfig["grounding_idim"] = mconfig["lgi_global_satt_att_edim"]
        else:
            mconfig["grounding_idim"] = mconfig["lgi_global_satt_att_cand_dim"]

        return config

    @staticmethod
    def dataset_specific_config_update(config, dset):
        mconfig = config["model"]
        # Query Encoder
        mconfig["query_enc_emb_idim"] = len(list(dset.wtoi.keys()))
        mconfig["loc_word_emb_vocab_size"] = len(list(dset.wtoi.keys()))
        return config
