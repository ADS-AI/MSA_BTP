from torchvision import transforms
import numbers
import numpy as np
import os
import csv
import random
from text.twitter import twitterDataset
from text.sst2 import sst2Dataset
from text.wikitext import wikitextDataset
from text.ag_news import agNewsDataset
from text.pubmed import pubmedDataset
from text.twitter_finance import twitterFinanceDataset
from text.yelp import yelpDataset
from text.CustomDataset import CustomDataset

'''dataset_to_modelfamily : dataset name to model family name'''
dataset_to_modelfamily = { 
    'imdb' : 'sentiment',
    'yelp' : 'sentiment',
    'sst2' : 'sentiment',
    'amazon' : 'sentiment',
    'twitter' : 'sentiment',
    'twitter_finance' : 'sentiment',
    
    'blog' : 'classification',
    'pubmed' : 'classification',
    'wiki_medical_terms' : 'classification',
}


def load_existing_dataset(dataset_name):
    if dataset_name == 'yelp':
        return yelpDataset
    elif dataset_name == 'sst2':
        return sst2Dataset
    elif dataset_name == 'pubmed':
        return pubmedDataset
    elif dataset_name == 'twitter_finance':
        return twitterFinanceDataset
    elif dataset_name == 'twitter':
        return twitterDataset
    elif dataset_name == 'ag_news':
        return agNewsDataset
    elif dataset_name == 'wiki_medical_terms':
        return wikitextDataset
    
def load_custom_dataset(path):
    '''
    Loads a custom dataset from the path
    '''
    # pass
    return CustomDataset
    
    



