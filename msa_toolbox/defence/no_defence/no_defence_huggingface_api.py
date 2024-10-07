import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from ...utils.image.cfg_reader import CfgNode
from ...utils.image.load_data_and_models import get_data_loader


'''
Function to be used when either:
Victim has no defense mechansim to use OR model stealing attack is not detected by the defense mechanism 
'''
def label_samples_with_no_defence_huggingface(cfg:CfgNode, thief_data:Dataset, 
            next_training_samples_indices:np.array, take_action:bool=False):
    '''
    Labels the new thief training samples using the victim model
    '''
    # addendum_loader = get_data_loader(Subset(thief_data, next_training_samples_indices), 
                        # batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    addendum_loader = DataLoader(Subset(thief_data, next_training_samples_indices), batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)
    print('Obtaining new labels for the thief data')
    
    for img, label0, index in addendum_loader:
        for ii, jj in enumerate(index): 
            image_path = thief_data.samples[jj][0]
            if cfg.TRAIN.BLACKBOX_TRAINING == True:
                label = call_victim_api_for_label(cfg, image_path, blackbox=True)
                thief_data.samples[jj] = (thief_data.samples[jj][0], label)
            else:
                label = call_victim_api_for_label(cfg, image_path, blackbox=False)
                thief_data.samples[jj] = (thief_data.samples[jj][0], label)
            print(label, end=' ')
        print()
    print('New labels for the thief data have been obtained')
    return


def call_victim_api_for_label(cfg:CfgNode, image_path, blackbox:bool=True):
    '''
    Calls the victim API to get the label for the image
    '''
    from PIL import Image
    from transformers import pipeline
    
    img = Image.open(image_path)
    # Change these to whatever model and image URL you want to use
    MODEL_ID = cfg.VICTIM.MODEL_ID

    # get the output response from the model
    classifier = pipeline('image-classification', model=MODEL_ID)
    output = classifier(img)
    output = list(output)  # Convert generator to list
    output = output if output is not None else []

    class_to_idx = {}
    probabilities = []
    for i in range(len(output)):
        class_label = output[i] 
        class_to_idx[class_label['label']] = len(class_to_idx)
        probabilities.append(class_label['score'])

    cfg.VICTIM.CLASS_TO_IDX = class_to_idx
    cfg.VICTIM.IDX_TO_CLASS = {v: k for k, v in class_to_idx.items()}
    cfg.VICTIM.NUM_CLASSES = len(class_to_idx)
    if blackbox == True:
        return torch.argmax(torch.tensor(probabilities)).item()
    else:
        return probabilities