import numbers
import numpy as np
import os
import csv
import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch

class Example(object):
    def __init__(self, guid, text_a, label=None, meta=None, att=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = None
        self.label = label
        self.att = att
        self.aux_label = []

        if meta is not None:
            if att is not None:
                for no in range(att):
                    if str(no) in meta:
                        self.aux_label.append("1")
                    else:
                        self.aux_label.append("0")
    def __str__(self):
        text = ""
        text += "guid: {}\n".format(self.guid)
        text += "text: {}\n".format(self.text_a)
        text += "label: {}".format(self.label)

        return text

# augmentaion left to do
class CustomDataset:
    
    def __init__(self, dataset_name, data_directory, num_labels=0, num_attributes=0, label_probs=False, data_import_method='tsv', config_file=None, tokenizer=None , seed = None):
        """
        Initialize a custom dataset instance.

        Args:
            dataset_name (str): Name of the dataset.
            data_directory (str): Directory containing the dataset files.
            num_labels (int): Number of labels in the dataset.
            num_attributes (int): Number of attributes in the dataset.
            label_probs (bool): Flag indicating whether label probabilities are used.
            data_import_method (str): Method used for importing data ('tsv', 'HuggingFace', etc.).
            config_file (object): config_file object.
            tokenizer (object): Tokenizer instance for text processing.
        """
        self.dataset_name = dataset_name
        self.data_directory = data_directory
        self.num_labels = num_labels
        self.num_attributes = num_attributes
        self.label_probs = label_probs
        self.data_import_method = data_import_method
        self.config_file = config_file
        self.tokenizer = tokenizer
        self.new_label = None
        self.seed = seed
        self.label_map = None

    def create_label_map(self):
        """
        Creates a label map for the dataset.
        """
        label_list = [str(i) for i in range(self.num_labels)]
        self.label_map = {label: i for i, label in enumerate(label_list)}
    
    def get_examples(self , split = "train"):
        '''
        Input:
            split: train, test or val
        Output:
            examples: list of InputExample
        function:
            get examples for split
        '''
        if self.data_import_method == 'tsv':
            file = split + '.tsv'
            return self._create_examples(self._read_tsv(os.path.join(self.data_directory, file)))

        elif self.data_import_method == 'csv':
            file = split + '.csv'
            return self._create_examples(self._read_csv(os.path.join(self.data_directory, file)))
        elif self.data_import_method == 'HuggingFace':
            print("Loading dataset from HuggingFace...")
            return self._create_examples(self._read_hugging(split = split))
    
    def set_new_labels(self, new_label):
        '''
        Input:
            new_label: new label
        function:
            set new label
        '''
        self.new_label = new_label
    
    def get_labels(self):
        """
        Gets the list of labels for the dataset.
        """
        return [str(i) for i in range(self.num_labels)]

    def _read_tsv(self, input_file, quotechar=None):
      """Reads a tab separated value file."""
      with open(input_file, "r") as f:
          reader = csv.reader((line.replace('\0','') for line in f), delimiter="\t", quotechar=quotechar)
          lines = []
          for line in reader:
              lines.append(line)
          return lines
      
    def _read_csv(self, input_file, quotechar=None):
        """
        Reads a comma separated value file.
        """
        with open(input_file, "r") as f:
            reader = csv.reader((line.replace('\0','') for line in f), delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines):
        examples = []
        # print(lines[0])
        for i, line in enumerate(lines):
            if len(line) == 1:
                examples.append(Example(i, line[0]))
            elif len(line) == 2:
                label = line[1].split()
                # assert len(label) == self.num_labels, "the number of labels does not match the predicted probs"
                if len(label) > 1:
                    examples.append(Example(i, line[0], [float(l) for l in label]))
                else:
                    examples.append(Example(i, line[0], label[0]))
            else:
                examples.append(Example(i, line[0], line[1], line[2], self.num_attributes))
        return examples

    def _read_hugging(self, split='train'):
        """
        will be inherited by each dataset class
        """
        raise NotImplementedError
         
    def get_features(self , split = "train", indexes = None, tokenizer = None, max_length=512, label_list=None, pad_on_left=False, pad_token=0, pad_token_segment_id=0, mask_padding_with_zero=True):
        '''
        Input:
            split: train, test or val
            indexes: list of indexes that we want to get features for them
            tokenizer: tokenizer(can be None or predefined during init)
            max_length: max length of input
            label_list: list of label
            pad_on_left: pad on left
            pad_token: pad token
            pad_token_segment_id: pad token segment id
            mask_padding_with_zero: mask padding with zero
        Output:
            features: list of InputFeatures
        function:
            convert examples to features for model
        '''
        if split == 'train':
            examples = self.train_example
        elif split == 'test':
            examples = self.test_example
        elif split == 'val':
            examples = self.val_example
        else:
            examples = None
            raise ValueError('split must be train, test or val')
            

        if tokenizer is None:
            tokenizer = self.tokenizer
        
        if label_list is not None:
            label_map = {label: i for i, label in enumerate(label_list)}
        else:
            label_map = self.create_label_map()
            # raise ValueError('label_list must be defined')

        aux_label_map = {"0": 0, "1": 1}

        if examples is not None:
            features = []
            for (ex_index, example) in enumerate(examples):
                inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, truncation=True)
                if "token_type_ids" in inputs:
                    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
                else:
                    input_ids, token_type_ids = inputs["input_ids"], None

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = max_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                    if token_type_ids is not None:
                        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
                assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
                if token_type_ids is not None:
                    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
                
                try:
                    label = label_map[example.label] if type(example.label) == str else example.label
                except:
                    print(example.label)
                    print(example.text_a)
                    print(example.text_b)
                    raise
                
                aux_label = [aux_label_map[l] for l in example.aux_label] if example.aux_label is not None else example.aux_label

                features.append(
                    InputFeatures(
                        guid=example.guid, 
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        token_type_ids=token_type_ids,
                        label=label, 
                        aux_label=aux_label)
                    )

            if indexes is not None:
                features = [features[i] for i in indexes]
    
        return features

    def get_loaded_features(self , index = None, split = "train" , true_labels = None, features = None):
        if features is None:
            if split == 'train':
                features = self.train_features
            elif split == 'test':
                features = self.test_features
        

        all_guids = []
        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_labels = []
        all_aux_labels = []
        if index is None:
            for i in range(len(features)):
                all_guids.append(features[i].guid)
                all_input_ids.append(features[i].input_ids)
                all_attention_mask.append(features[i].attention_mask)
                if features[i].token_type_ids is not None:
                    all_token_type_ids.append(features[i].token_type_ids)
                else:
                    all_token_type_ids.append(0)
                if true_labels is None:
                    if features[i].label is None:
                        all_labels.append(0)
                    else:
                        all_labels.append(features[i].label)
                else:
                    all_labels.append(true_labels[i])
                all_aux_labels.append(features[i].aux_label)

        else:
            for i in range(len(features)):
                if i in index:
                    all_guids.append(features[i].guid)
                    all_input_ids.append(features[i].input_ids)
                    all_attention_mask.append(features[i].attention_mask)
                    if features[i].token_type_ids is not None:
                        all_token_type_ids.append(features[i].token_type_ids)
                    else:
                        all_token_type_ids.append(0)
                    if true_labels is None:
                        if features[i].label is None:
                            all_labels.append(0)
                        else:
                            all_labels.append(features[i].label)
                    else:
                        all_labels.append(true_labels[i])
                    all_aux_labels.append(features[i].aux_label)
    
        all_guids = torch.tensor(all_guids, dtype=torch.long)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
        if type(all_labels[0]) == list:
            all_labels = torch.tensor(all_labels,  dtype=torch.float64)
        else:
            all_labels = torch.tensor(all_labels,  dtype=torch.long)
        all_aux_labels = [torch.tensor(all_aux_labels, dtype=torch.long) for i in range(len(features[0].aux_label))] 
        dataset = TensorDataset(all_guids, all_input_ids, all_attention_mask, all_token_type_ids, all_labels, *all_aux_labels)
        return dataset


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, guid, input_ids, attention_mask=None, token_type_ids=None, label=None, aux_label=None):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.aux_label = aux_label