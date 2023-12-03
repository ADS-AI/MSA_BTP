import os
from .CustomDataset import CustomDataset
from datasets import load_dataset

# inherits ecerything from custom dataset + some dataset specific functions
class twitterDataset(CustomDataset):

    def __init__(self, dataset_name = "twitter", data_directory = None , num_labels=2, num_attributes=0, label_probs=False, data_import_method='HuggingFace',  config_file=None, tokenizer=None , seed = None, type = ""):
        super().__init__(dataset_name, data_directory, num_labels, num_attributes, label_probs, data_import_method, config_file, tokenizer , seed)
        self.train_example = self.get_examples(split = 'train')
        self.test_example = self.get_examples(split = 'test')
        self.train_features = None
        self.test_features = None

    def _read_hugging(self ,  split = 'train'):
        lines = []
        dataset = load_dataset('carblacac/twitter-sentiment-analysis', split = split)
        for row in dataset:
            lines.append([row["text"],  str(row['feeling'])]) # type: ignore

        return lines
