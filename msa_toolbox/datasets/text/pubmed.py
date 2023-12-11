import os
from .CustomDataset import CustomDataset
from datasets import load_dataset

# inherits ecerything from custom dataset + some dataset specific functions
class pubmedDataset(CustomDataset):

    def __init__(self, dataset_name = "pubmed", data_directory = None, num_labels=5, num_attributes=0, label_probs=False, data_import_method='HuggingFace',  config_file=None, tokenizer=None , seed = None):
        super().__init__(dataset_name, data_directory, num_labels, num_attributes, label_probs, data_import_method, config_file, tokenizer , seed)
        self.train_example = self.get_examples(split = 'train')
        self.test_example = self.get_examples(split = 'test')
        self.train_features = None
        self.test_features = None
        
    def _read_hugging(self ,  split = 'train'):
        lines = []
        print("Downloading dataset..." , split)
        dataset = load_dataset('ml4pubmed/pubmed-text-classification-cased', split = split)
            # create label map
        if self.label_map is None:
            self.create_label_map()
        for i in range(len(dataset)):
                test = dataset[i]['description']
                label = dataset[i]['target']
                map_label = str(self.label_map[label])
                lines.append([test , map_label])
        return lines

    def create_label_map(self):
        labels = ['BACKGROUND', 'RESULTS', 'METHODS', 'CONCLUSIONS', 'OBJECTIVE']
        self.label_map = {}
        for (i, label) in enumerate(labels):
            self.label_map[label] = i
