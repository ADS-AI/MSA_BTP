import argparse
import os
import sys
import logging
import torch
import random
from datetime import datetime
import yaml
import torch.nn as nn
import numpy as np


def load_yaml(path):
    '''
    Loads a YAML file and returns a dictionary.
    Arguments:
        - path (str): The path to the YAML file.
    '''
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class CfgNode:
    '''
    Class to convert a dictionary to an CFG object.
    Arguments:
        - dictionary (dict): The dictionary to convert to an CFG object.
    '''

    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                v = CfgNode(v)
            setattr(self, k, v)

    def __repr__(self):
        '''
        String representation of the CFG object.
        '''
        return str(self.__dict__)


def load_cfg(path):
    '''
    Loads a YAML file and returns a CfgNode object.
    Arguments:
        - path (str): The path to the YAML file.
    '''
    cfg = load_yaml(path)
    cfg = CfgNode(cfg)
    cfg.INTERNAL_LOG_PATH = './msa_toolbox/ui_flask/logs/'
    cfg.ACTIVE.VAL = cfg.ACTIVE.BUDGET // (2 * cfg.ACTIVE.CYCLES)
    rest_samples = cfg.ACTIVE.BUDGET - cfg.ACTIVE.VAL
    cfg.ACTIVE.INITIAL = rest_samples // cfg.ACTIVE.CYCLES
    cfg.ACTIVE.ADDENDUM = cfg.ACTIVE.INITIAL 
    return cfg
