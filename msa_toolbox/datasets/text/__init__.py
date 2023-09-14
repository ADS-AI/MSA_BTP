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
