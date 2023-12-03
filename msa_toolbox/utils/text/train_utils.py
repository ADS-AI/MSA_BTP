import os
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from . cfg_reader import CfgNode
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from torch.optim import Adam, AdamW
from scipy.stats import entropy
from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import pdb

def set_seed(cfg , n_gpu):
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(cfg.SEED)
        
def simple_accuracy(preds, labels):
        return (preds == labels).mean()

def agreement_score(thief_labels: list[int], victim_labels: list[int]) -> float:
    '''
    Calculates the agreement between the thief and victim model based on output label lists
    '''
    assert len(thief_labels) == len(victim_labels), "Label lists must have the same length"
    
    correct = sum(thief_label == victim_label for thief_label, victim_label in zip(thief_labels, victim_labels))
    total_samples = len(thief_labels)
    
    return correct / total_samples

def evaluate(cfg, DataLoader, model, tokenizer, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model.to(device)
    print("***** Running evaluation *****")
    results = {}
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(DataLoader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[1], "attention_mask": batch[2], "labels": batch[4]}
            if cfg.THIEF.ARCHITECTURE != "distilbert":
                inputs["token_type_ids"] = (batch[3] if cfg.THIEF.ARCHITECTURE in ["bert", "dpbert", "xlnet", "albert", "dstilbert"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy() , axis=0)
    nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    results = {"acc": simple_accuracy(preds, out_label_ids)}
    return results, preds





def active_train(cfg , train_dataloader , eval_dataloader , model , config , tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model.to(device)

    # optimizer and scheduler
    if cfg.THIEF_HYPERPARAMETERS.MAX_STEPS > 0:
        t_total = cfg.THIEF_HYPERPARAMETERS.MAX_STEPS
        cfg.THIEF_HYPERPARAMETERS.EPOCH = cfg.THIEF_HYPERPARAMETERS.MAX_STEPS // (len(train_dataloader) // cfg.THIEF_HYPERPARAMETERS.GRAD_ACCUM_STEPS) + 1
    else:
        t_total = len(train_dataloader) // cfg.THIEF_HYPERPARAMETERS.GRAD_ACCUM_STEPS * cfg.THIEF_HYPERPARAMETERS.EPOCH
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.THIEF_HYPERPARAMETERS.WDECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    if(cfg.THIEF_HYPERPARAMETERS.OPTIMIZER == "Adam"):
        learning_rate = cfg.THIEF_HYPERPARAMETERS.LR
        # pdb.set_trace()
        # learning_rate = float(learning_rate)
        optimizer = AdamW(optimizer_grouped_parameters, lr= float(cfg.THIEF_HYPERPARAMETERS.LR), eps= float(cfg.THIEF_HYPERPARAMETERS.ADAM_EPS))
    scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=cfg.THIEF_HYPERPARAMETERS.WARMUP_STEPS, num_training_steps=t_total)


    # Train
    print("***** Running training *****")
    print("  Num examples = ", len(train_dataloader) , "  Num Epochs = ", cfg.THIEF_HYPERPARAMETERS.EPOCH, "\n")
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(cfg.THIEF_HYPERPARAMETERS.EPOCH), desc="Epoch", disable=cfg.LOCAL_RANK not in [-1, 0])
    set_seed(cfg, n_gpu) 
    for main_epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=cfg.LOCAL_RANK not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[1], "attention_mask": batch[2], "labels": batch[4]}
            if cfg.THIEF.ARCHITECTURE != "distilbert":
                inputs["token_type_ids"] = (
                        batch[3] if cfg.THIEF.ARCHITECTURE in ["bert", "dpbert", "xlnet", "albert", "dstilbert"] else None
                ) 
            outputs = model(**inputs)
            loss = outputs[0]  
            if n_gpu > 1:
                loss = loss.mean()
            if cfg.THIEF_HYPERPARAMETERS.GRAD_ACCUM_STEPS > 1:
                loss = loss / cfg.THIEF_HYPERPARAMETERS.GRAD_ACCUM_STEPS

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) %  cfg.THIEF_HYPERPARAMETERS.GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.THIEF_HYPERPARAMETERS.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step() 
                model.zero_grad()
                global_step += 1
        current_epoch = main_epoch
        # if current_epoch % cfg.TRAIN.EVAL_EVERY == 0:
        #     print("***** Running evaluation *****")
        #     results = evaluate(cfg, eval_dataloader, model, tokenizer, config)
        #     with open(os.path.join(cfg.THIEF_MODEL_DIR, "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.ARCHITECTURE , cfg.ACTIVE.METHOD), "prefix-{}".format(prefix)  , "cycle_num-{}".format(cycle_num) , "epoch-{}".format(current_epoch) , "eval_results.txt"), "w") as writer :
        #         writer.write("Eval Accuracy - {}\n".format(results["acc"]))
        #         writer.write("Train_examples - {}\n".format(len(train_dataloader)))
        #         writer.write("Train_examples - {}\n".format(len(eval_dataloader)))
    epoch_iterator.close()
    train_iterator.close()

    

            
    acc, preds = evaluate(cfg, eval_dataloader, model, tokenizer, config)

        
    eval_results = {}
    eval_results["acc"] = acc
    return global_step, tr_loss / global_step , eval_results , model , config , tokenizer 


def query_data_in_victim(cfg , theif_dataloader, victim_model, victim_tokenizer, victim_config):
    '''
    This function is used to query all the data from the victim model
    returns- 
    index_to_new_label - dictionary mapping index to new label
    index_to_entropy -  dictionary mapping index to entropy
    orignal_index - list of original index
    '''
    print("--------------- victim model setup for querying ---------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    victim_model.to(device)
    
    print("--------------- querying ---------------")
    new_labels = None
    orignal_labels = None
    orignal_index = None
    entropy = None
    for batch in tqdm(theif_dataloader, desc="Evaluating"):
        victim_model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[1], "attention_mask": batch[2]}
            outputs = victim_model(**inputs)
            logits = outputs['logits']
        if new_labels is None:
            new_labels = logits.detach().cpu().numpy()
            orignal_labels = batch[4].detach().cpu().numpy()
            orignal_index = batch[0].detach().cpu().numpy()
            entropy = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            entropy = np.sum(entropy * np.log(entropy), axis=1)
        else:
            new_labels = np.append(new_labels, logits.detach().cpu().numpy(), axis=0)
            orignal_labels = np.append(orignal_labels, batch[4].detach().cpu().numpy(), axis=0)
            orignal_index = np.append(orignal_index, batch[0].detach().cpu().numpy(), axis=0)
            tmp_entropy = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            tmp_entropy = np.sum(tmp_entropy * np.log(tmp_entropy), axis=1)
            entropy = np.append(entropy, tmp_entropy, axis=0)

    new_labels = np.argmax(new_labels, axis=1)
    index_to_new_label = {}
    index_to_entropy = {}

    for i in range(len(orignal_index)):
        index_to_new_label[orignal_index[i]] = new_labels[i]
        index_to_entropy[orignal_index[i]] = entropy[i]

    print("--------------- querying done ---------------")
    return index_to_new_label, index_to_entropy , orignal_index

   
def get_entropy(cfg , model , tokenizer , config , dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    entropy = None
    orignal_index = None
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[1], "attention_mask": batch[2]}
            outputs = model(**inputs)
            logits = outputs['logits']
            # print("logits", logits)
        if entropy is None:
            orignal_index = batch[0].detach().cpu().numpy()
            entropy = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            entropy = np.sum(entropy * np.log(entropy), axis=1)
        else:
            orignal_index = np.append(orignal_index, batch[0].detach().cpu().numpy(), axis=0)
            tmp_entropy = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            tmp_entropy = np.sum(tmp_entropy * np.log(tmp_entropy), axis=1)
            entropy = np.append(entropy, tmp_entropy, axis=0)
    
    index_to_entropy = {}
    for i in range(len(orignal_index)):
        index_to_entropy[orignal_index[i]] = entropy[i]
    
    return index_to_entropy


def get_label_probs(cfg , model , tokenizer , config ,  dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    label_probs = None
    orignal_index = None
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[1], "attention_mask": batch[2]}
            outputs = model(**inputs)
            logits = outputs['logits']
            # print("logits", logits)
        if label_probs is None:
            orignal_index = batch[0].detach().cpu().numpy()
            probability = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            label_probs = probability
            
        else:
            orignal_index = np.append(orignal_index, batch[0].detach().cpu().numpy(), axis=0)
            tmp_probability = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            label_probs = np.append(label_probs, tmp_probability, axis=0)
    
    index_to_label_probs = {}
    for i in range(len(orignal_index)):
        index_to_label_probs[orignal_index[i]] = label_probs[i]
    
    return index_to_label_probs

def get_index_by_vote(index_to_label_prob):
    votes = []
    # pdb.set_trace()
    total_keys = len(index_to_label_prob.keys())
    for key in index_to_label_prob.keys():
        labels = []
        for i in index_to_label_prob[key].keys():
            label = np.argmax(index_to_label_prob[key][i])
            labels.append(label)
        votes.append(labels)

    entropy_list  = []
    for j in range(len(votes)):
        probs = []
        for i in range(total_keys):
            i_count = votes[j].count(i)
            probs.append(i_count/len(votes[j]))
        entropy_list.append(entropy(probs))

    entropy_list_sorted = sorted(entropy_list, reverse=True)
    return entropy_list_sorted

def get_index_by_disagreement(index_to_label_prob):
    disagreements = []
    total_keys = len(index_to_label_prob.keys())
    
    for key in index_to_label_prob.keys():
        labels = [np.argmax(index_to_label_prob[key][i]) for i in index_to_label_prob[key].keys()]
        disagreement = 1.0 - (np.max(np.bincount(labels)) / len(labels))
        disagreements.append(disagreement)

    disagreements_sorted = sorted(enumerate(disagreements), key=lambda x: x[1], reverse=True)
    return disagreements_sorted
