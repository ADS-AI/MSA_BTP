import os
from CustomDataset import CustomDataset
from datasets import load_dataset

# inherits ecerything from custom dataset + some dataset specific functions
class yelpDataset(CustomDataset):

    def __init__(self, dataset_name, data_directory, num_labels=5, num_attributes=0, label_probs=False, data_import_method='HuggingFace',  config_file=None, tokenizer=None , seed = None):
        super().__init__(dataset_name, data_directory, num_labels, num_attributes, label_probs, data_import_method, config_file, tokenizer , seed)

    def _read_hugging(self ,  split = 'train'):
        lines = []
        dataset = load_dataset('yelp_review_full', split = split)
            # create label map
        for i in range(len(dataset)):
                test = dataset[i]['text']
                label = dataset[i]['label']
                lines.append([test , label])
        return lines
