import torch.nn as nn

# import networks
from src.model import LGI

""" Network Mapping """
TGN_LGI = LGI.LGI

def get_rnn(config, prefix=""):
    name = prefix if prefix is "" else prefix+"_"

    # fetch options
    rnn_type = config.get(name+"rnn_type", "LSTM")
    bidirectional = config.get(name+"rnn_bidirectional", True)
    nlayers = config.get(name+"rnn_nlayer", 2)
    idim = config.get(name+"rnn_idim", -1)
    hdim = config.get(name+"rnn_hdim", -1)
    dropout = config.get(name+"rnn_dropout", 0.5)

    rnn = getattr(nn, rnn_type)(idim, hdim, nlayers,
                                batch_first=True, dropout=dropout,
                                bidirectional=bidirectional)
    return rnn

def get_rnn_cell(config, prefix=""):
    name = prefix if prefix is "" else prefix+"_"

    # fetch options
    idim = config.get(name+"cell_idim", 500)
    hdim = config.get(name+"cell_hdim", 512)
    cell_type = config.get(name+"cell_type", "GRUCell")

    cell = getattr(nn, cell_type)(idim, hdim)
    return cell

def get_temporal_grounding_network(config, net_type="tgn", raw=False):
    if net_type == "tgn_lgi":
        M = TGN_LGI
    else:
        raise NotImplementedError(
            "Not supported TGN ({})".format(net_type))

    if raw: return M
    return M(config)
