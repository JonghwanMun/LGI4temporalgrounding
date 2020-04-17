import os
import csv
import json
import yaml
import logging, logging.handlers
from six.moves import cPickle as pickle

import h5py
import coloredlogs
import numpy as np
from collections import defaultdict


""" Get Logger with given name
Args:
    name: logger name.
    fmt: log format. (default: %(asctime)s %(name)s: %(message)s)
    level: logging level. (default: logging.DEBUG)
    log_file: path of log file. (default: None)
"""
def get_logger(name, log_file_path=None, fmt="%(asctime)s %(name)s: %(message)s",
               print_lev=logging.DEBUG, write_lev=logging.INFO):
    logger = logging.getLogger(name)
    # Add file handler
    if log_file_path:
        formatter = logging.Formatter(fmt)
        file_handler = logging.handlers.RotatingFileHandler(log_file_path)
        file_handler.setLevel(write_lev)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # Add stream handler
    coloredlogs.install(level=print_lev, logger=logger,
                        fmt="%(asctime)s %(name)s %(message)s")
    return logger


""" pickle helpers """
def load_pkl(file_path, verbose=True):
    with open(file_path, "rb") as f:
        pkl_file = pickle.load(f)
    if verbose:
        print("Load pkl file from {}".format(file_path))
    return pkl_file

def write_pkl(file_path, data, verbose=True):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    if verbose:
        print("Write pkl file in {}".format(file_path))


""" YAML helpers """
def load_yaml(file_path, verbose=True):
    with open(file_path, "r") as f:
        yml_file = yaml.load(f, Loader=yaml.SafeLoader)
    if verbose:
        print("Load yaml file from {}".format(file_path))
    return yml_file

def write_yaml(file_path, yaml_data, verbose=True):
    with open(file_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    if verbose:
        print("Write yaml file in {}".format(file_path))

""" Text file helpers """
def append_text_to_file(file_path, append):
    """ Append text to the file located in file_path.
    Args:
        file_path: str, file path
        append: str, text to append
    Return: None
    """
    with open(file_path, "a") as f:
        if append[-1] == "\n":
            append = append[:-1]
        f.write(append + "\n")

def load_lines_from(file_path):
    lines = []
    f = open(file_path, "r")
    while True:
        line = f.readline()
        if not line: break
        lines.append(line.strip().strip("\n"))
    f.close()
    return lines

def load_lines_as_dict(file_path):
    dic = defaultdict(lambda: False)
    f = open(file_path, "r")
    while True:
        line = f.readline()
        if not line: break
        dic[line.strip().strip("\n")] = True
    f.close()
    return dic

""" csv helpers """
def load_csv(file_path):
    out = []
    with open(file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out

""" JSON helpers """
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, \
                            np.int16, np.int32, np.int64, np.uint8, \
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def write_json(file_path, file_data, verbose=True):
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        else: return json.JSONEncoder.default(o)

    with open(file_path, "w") as outfile:
        json.dump(file_data, outfile, default=default)

    if verbose:
        print("Write json file in {}".format(file_path))

def load_json(file_path, verbose=True):
    if verbose:
        print("Load json file from {}".format(file_path))
    return json.load(open(file_path, "r"))

""" HDF5 helpers """
def open_hdf5(file_path, mode="r", verbose=True):
    if verbose:
        print("Open hdf5 file from {}".format(file_path))
    return h5py.File(file_path, mode)

def load_hdf5(file_path, verbose=True):
    if verbose:
        print("Load hdf5 file from {}".format(file_path))
    return h5py.File(file_path, "r")

def load_hdf5_as_numpy_array_dict(file_path, target_group=None):
    f = h5py.File(file_path, "r")
    if target_group is None:
        hdf5_dict = {k: np.array(v) for k, v in f.items()}
    else:
        hdf5_dict = {k: np.array(v) for k, v in f[target_group].items()}
    print("Load hdf5 file: {}".format(file_path))
    return hdf5_dict

def print_hdf5_keys(hdf5_file):
    def printname(name):
        print (name)
    hdf5_file.visit(printname)


""" Directory helpers """
def check_and_create_dir(dir_path):
    if not os.path.isdir(dir_path):
        print("Create directory: {}".format(dir_path))
        os.makedirs(dir_path, exist_ok=True)

def get_filenames_from_dir(search_dir_path, all_path=False):
    """ Get filename list from a directory recursively
    """

    filenames = []
    dirs = []
    for (path, dr, files) in os.walk(search_dir_path):
        dirs.append(path)
        for f in files:
            filenames.append(os.path.join(path if all_path else path.replace(search_dir_path, ""), f))
    return filenames, dirs

