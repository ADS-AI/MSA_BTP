"""
This module defines the CUBS200 dataset class. It provides a wrapper around the CUB-200-2011 dataset
for image classification. The dataset contains 11,788 images of 200 bird species,
with each species having 30 images. 

Note that the CUB-200-2011 dataset must be downloaded separately and placed in the specified root
directory before using this module. 
The download link can be found at http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.
"""

import os
from torchvision.datasets.folder import ImageFolder
import msa_toolbox.config as cfg


class CUBS200(ImageFolder):
    """
    CUBS200 dataset class, subclass of ImageFolder.

    Attributes:
        root (str): Root directory path.
        partition_to_idxs (dict): Mapping of filenames to train or test partition.
        pruned_idxs (list): List of indices of samples in the required train or test partition.
    
    Methods:
        __init__(self, train=True, transform=None, target_transform=None):
            Initializes the CUBS200 dataset.
        get_partition_to_idxs(self) -> Dict[str, List[int]]:
            Returns a mapping of filenames to train or test partition.
    """

    def __init__(self, train=True, transform=None, target_transform=None):
        """
        Initializes the CUBS200 dataset.

        Args:
            train (bool, optional): Whether to load the training partition or testing partition.
                    Defaults to True.
            transform (callable, optional): A function/transform that takes in a sample and 
                    returns a transformed version. Defaults to None.
            target_transform (callable, optional): A function/transform that takes in the target
                    and transforms it. Defaults to None.
        
        Raises:
            ValueError: If dataset is not found at the root directory path.
        """
        root = os.path.join(cfg.DATASET_ROOT, 'CUB_200_2011')
        if not os.path.exists(root):
            raise ValueError(f'Dataset not found at {root}. Please download it from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.')

        super().__init__(root=os.path.join(root, 'images'), transform=transform,
                        target_transform=target_transform)
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print(f"=> done loading {self.__class__.__name__} " f"({'train' if train else 'test'}) with {len(self.samples)} examples")


    def get_partition_to_idxs(self):
        """
        Creates a mapping of each image's partition (train or test) to its corresponding indices in the dataset.

        Returns:
            A dictionary containing two keys, 'train' and 'test', with values being lists of indices corresponding to
            the train and test partitions, respectively.
        """
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # ----------------- Create mapping: filename -> 'train' / 'test'
        # There are two files: a) images.txt containing: <imageid> <filepath>
        #            b) train_test_split.txt containing: <imageid> <0/1>
        # First, create a mapping from image ID to filename using the images.txt file.
        imageid_to_filename = dict()
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                imageid, filepath = line.strip().split()
                _, filename = os.path.split(filepath)
                imageid_to_filename[imageid] = filename

        # Then, create a mapping from filename to image ID by reversing the imageid_to_filename mapping.
        filename_to_imageid = {v: k for k, v in imageid_to_filename.items()}

        # Next, create a mapping from image ID to partition (train or test) using the 
        # train_test_split.txt file.
        imageid_to_partition = dict()
        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            for line in f:
                imageid, split = line.strip().split()
                imageid_to_partition[imageid] = 'train' if int(split) else 'test'

        # Finally, loop through each sample in the dataset and group them based on their partition.
        for idx, (filepath, _) in enumerate(self.samples):
            _, filename = os.path.split(filepath)
            imageid = filename_to_imageid[filename]
            partition_to_idxs[imageid_to_partition[imageid]].append(idx)

        return partition_to_idxs
