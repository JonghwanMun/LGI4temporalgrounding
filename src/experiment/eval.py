import json
import argparse

from src.experiment import common_functions as cmf
from src.utils import io_utils

""" Get parameters """
def _get_argument_params():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True,
                        help="Experiment or configuration name")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="filename of checkpoint.")
    parser.add_argument("--method", default="tgn_lgi",
                        help="Method type")
    parser.add_argument("--dataset", default="charades",
                        help="dataset to train models [charades|anet].")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="The number of workers for data loader.")
    parser.add_argument("--debug_mode" , action="store_true", default=False,
                        help="Run the script in debug mode")

    params = vars(parser.parse_args())
    print (json.dumps(params, indent=4))
    return params

def main(params):
    config = io_utils.load_yaml(params["config"])


    # prepare dataset
    D = cmf.get_dataset(params["dataset"])
    dsets, L = cmf.get_loader(D, split=["test"],
                              loader_configs=[config["test_loader"]],
                              num_workers=params["num_workers"])

    # Build network
    M = cmf.get_method(params["method"])
    net = M(config, logger=None)
    net.load_checkpoint(params["checkpoint"], True)
    if config["model"]["use_gpu"]: net.gpu_mode()

    # Evaluating networks
    cmf.test(config, L["test"], net, -1, None, mode="Test")

if __name__ == "__main__":
    params = _get_argument_params()
    main(params)
