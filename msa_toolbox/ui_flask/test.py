import yaml
import os
import random
import json
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, jsonify

def chart():
    # create some random data for demonstration purposes
    # labels = ["January", "February", "March", "April", "May", "June", "July"]
    
    metric_path = 'msa_toolbox/ui_flask/logs/log_metrics.json'
    with open(metric_path) as f:
        data = json.load(f)
    accuracy_victim = []
    accuracy_thief = []
    
    precision_victim = []
    precision_thief = []
    
    f1_victim = []
    f1_thief = []
    
    agreement_victim = []
    agreement_thief = []
    for i in data.keys():
        for j in data[i].keys():
            if j == 'metrics_victim':
                accuracy_victim.append(data[i][j]['accuracy'])
                precision_victim.append(data[i][j]['precision'])
                f1_victim.append(data[i][j]['f1'])
            elif j == 'agreement_victim':
                agreement_victim.append(data[i][j])
            elif j == 'metrics_thief':
                accuracy_thief.append(data[i][j]['accuracy'])
                precision_thief.append(data[i][j]['precision'])
                f1_thief.append(data[i][j]['f1'])
            elif j == 'agreement_thief':
                agreement_thief.append(data[i][j])
    print(accuracy_victim)
    print(accuracy_thief)
    print(precision_victim)
    print(precision_thief)
        # metric[i] = data[str(i)]['val_acc']

# chart()

def get_file_progress():
    # relative path to a file 
    # path1 = 'msa_toolbox/ui_flask/logs/log_tqdm.txt'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'logs/log_tqdm.txt')
    print(current_dir)
    # print(os.getcwd())  
    # path1 = os.path.join(os.getcwd(), 'msa_toolbox/ui_flask/logs/log_tqdm.txt')
    # print(path1)
    content = ''
    # # # path  = '/logs/log_tqdm.txt'
    with open(path) as f:
        
        content = f.read()
        # parse content to calculate progress
        # progress = content
        # print(progress)
    return content
    
print(get_file_progress())