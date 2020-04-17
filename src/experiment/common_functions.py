import os
import logging
from tqdm import tqdm

import torch

from src.dataset import anet, charades
from src.model import building_networks as bn
from src.utils import utils, io_utils

def get_method(method_type):
    if method_type.startswith("tgn"):
        M = bn.get_temporal_grounding_network(None, method_type, True)
    else:
        raise NotImplementedError("Not supported model type ({})".format(method_type))
    return M

def get_dataset(dataset):
    if dataset == "anet":
        D = eval("anet")
    elif dataset == "charades":
        D = eval("charades")
    else:
        raise NotImplementedError("Not supported dataset type ({})".format(dataset))
    return D

def get_loader(D, split=[], loader_configs=[], num_workers=2):
    assert len(split) > 0
    assert len(split) == len(loader_configs)
    return D.create_loaders(split, loader_configs, num_workers)

def update_config_from_params(config, params):
    config["misc"]["debug"] = params.get("debug_mode", False)
    config["misc"]["num_workers"] = params.get("num_workers", 0)
    config["misc"]["dataset"] = params["dataset"]
    exp_prefix = utils.get_filename_from_path(
            params["config_path"], delimiter="options/") if "options" in params["config_path"] \
            else utils.get_filename_from_path(params["config_path"], delimiter="results/")[:-7]
    config["misc"]["exp_prefix"] = exp_prefix
    config["misc"]["result_dir"] = os.path.join("results", exp_prefix)
    config["misc"]["tensorboard_dir"] = os.path.join("tensorboard", exp_prefix)
    config["misc"]["method_type"] = params["method_type"]
    if not "use_gpu" in config["model"].keys():
        if torch.cuda.is_available():
            config["model"]["use_gpu"] = True
        else:
            config["model"]["use_gpu"] = False

    return config

def prepare_experiment(params, update_config=True):
    M = get_method(params["method_type"])
    D = get_dataset(params["dataset"])

    # loading configuration and setting environment
    config = io_utils.load_yaml(params["config_path"])
    if update_config:
        config = update_config_from_params(config, params)
        create_save_dirs(config["misc"])

    return M, D, config

def factory_model(config, M, dset=None, logger=None):
    if dset is not None:
        config = M.dataset_specific_config_update(config, dset)
    net = M(config, logger=logger, verbose=True)
    if dset is not None:
        net.bring_dataset_info(dset)

    # load checkpoint
    s_iter = 1
    if config["model"]["resume"]:
        ckpt_path = config["model"]["checkpoint_path"]
        assert len(ckpt_path) > 0
        net.load_checkpoint(ckpt_path, True)
        if "epoch" in ckpt_path or "iter" in ckpt_path:
            if "epoch" in ckpt_path: d = "epoch"
            else: d = "iter"
            s_iter = int(utils.get_filename_from_path(
                    config["model"]["checkpoint_path"]).split(d)[-1])
            if "iter" in ckpt_path:
                net.it = s_iter

    # ship network to use gpu
    if config["model"]["use_gpu"]: net.gpu_mode()
    if logger is not None: logger.info(net)
    return net, s_iter

def create_save_dirs(config):
    """ Create neccessary directories for training and evaluating models
    """
	# create directory for checkpoints
    io_utils.check_and_create_dir(os.path.join(config["result_dir"], "checkpoints"))
	# create directory for results
    io_utils.check_and_create_dir(os.path.join(config["result_dir"], "status"))
    io_utils.check_and_create_dir(os.path.join(config["result_dir"], "qualitative"))


""" Get logger """
def create_logger(config, logger_name, log_path):
    logger_path = os.path.join(
            config["misc"]["result_dir"], log_path)
    logger = io_utils.get_logger(
        logger_name, log_file_path=logger_path,\
        print_lev=getattr(logging, config["logging"]["print_level"]),\
        write_lev=getattr(logging, config["logging"]["write_level"]))
    return logger

""" evaluate the network """
def test(config, loader, net, epoch, eval_logger=None, mode="Test"):

    with torch.no_grad():
        net.eval_mode() # set network as evaluation mode
        net.reset_status() # reset status
        net.reset_counters()

        # Testing network
        ii = 1
        for batch in tqdm(loader, desc="{}".format(mode)):
            # forward the network
            net_inps, gts = net.prepare_batch(batch)
            outputs = net.forward_only(net_inps) # only forward

            # Compute status for current batch: loss, evaluation scores, etc
            net.compute_status(outputs["net_output"], gts)

            ii += 1
            if config["misc"]["debug"] and (ii > 3):
                break
            # end for batch in loader

        net.save_results("epoch{:03d}".format(epoch), mode=mode)
        if epoch > 0 and net.renew_best_score():
            ckpt_path = os.path.join(config["misc"]["result_dir"], "checkpoints")
            net.save_checkpoint(os.path.join(
                    ckpt_path, "epoch{:03d}.pkl".format(epoch)))
            net.save_checkpoint(os.path.join(
                    ckpt_path, "best.pkl".format(epoch)))
        net.print_counters_info(eval_logger, epoch, mode=mode)

""" miscs """
def extract_output(config, loader, net, save_dir):

    with torch.no_grad():
        net.eval_mode() # set network as evaluation mode
        net.reset_status() # reset status
        net.reset_counters()

        # Testing network
        ii = 1
        for batch in tqdm(loader, desc="extract_output"):
            # forward the network
            net_inps, gts = net.prepare_batch(batch)
            outputs = net.extract_output(net_inps, gts, save_dir) # only forward

            ii += 1
            if config["misc"]["debug"] and (ii > 3):
                break
            # end for batch in loader

""" Methods for debugging """
def one_step_forward(L, net, logger):
    # fetch the batch
    batch = next(iter(L))

    # forward and update the network
    outputs = net.forward_update(batch)

    # accumulate the number of correct answers
    net.compute_status(outputs, batch["gt"])

    # print learning status
    net.print_status(1, logger)
