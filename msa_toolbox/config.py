import os
import os.path as osp
from os.path import dirname, abspath

DEFAULT_SEED = 42
DS_SEED = 123  # uses this seed when splitting datasets

# -------------- Paths
CONFIG_PATH = abspath(__file__)
SRC_ROOT = dirname(CONFIG_PATH)
PROJECT_ROOT = dirname(SRC_ROOT)
CACHE_ROOT = osp.join(SRC_ROOT, 'cache')
DATASET_ROOT = osp.join(SRC_ROOT, 'data')
DEBUG_ROOT = osp.join(SRC_ROOT, 'debug')
MODEL_DIR = osp.join(SRC_ROOT, 'models')

os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(DATASET_ROOT, exist_ok=True)
os.makedirs(DEBUG_ROOT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# print(CONFIG_PATH)
# print(SRC_ROOT)
# print(PROJECT_ROOT)
# print(CACHE_ROOT)
# print(DATASET_ROOT)
# print(DEBUG_ROOT)
# print(MODEL_DIR)

# -------------- Dataset Stuff
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_BATCH_SIZE = 64