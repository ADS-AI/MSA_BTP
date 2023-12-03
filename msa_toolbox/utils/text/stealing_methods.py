import os
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from .load_data_and_models import load_untrained_model, load_dataset_thief
from . cfg_reader import CfgNode
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from .load_victim_thief_data_and_model import save_thief_model
from .train_utils import query_data_in_victim, active_train , evaluate, get_entropy, set_seed, agreement_score, get_label_probs, get_index_by_vote
from ...utils.text.load_data_and_models import load_untrained_model
from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from ...models.text.model import BertForSequenceClassificationDistil
import pdb
import copy

# MODEL_CLASSES = {
#     "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
#     "dstilbert": (BertConfig, BertForSequenceClassificationDistil, BertTokenizer),
#     "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
#     "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
#     "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
#     "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
#     "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
#     "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
#     "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
# }


def active_learning_technique(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, thief_model, thief_tokenizer, thief_config ):
    # if cfg.ACTIVE.METHOD == "qbc_stealing":
    #     return qbc_stealing(cfg,None, thief_model)
    if cfg.ACTIVE.METHOD == "entropy_stealing":
        return entropy_stealing(cfg, victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config)
    elif cfg.ACTIVE.METHOD == "all_data_stealing":
        return all_data_stealing(cfg, victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config)
    elif cfg.ACTIVE.METHOD == "qbc_stealing":
        thief_list_models = {}
        thief_list_configs = {}
        thief_list_tokenizers = {}
        models = cfg.ACTIVE.MODELS
        for mod in models:
            thief_list_models[mod] , thief_list_tokenizers[mod] , thief_list_configs[mod] = load_untrained_model(cfg, mod)
        qbc(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, thief_list_models, thief_list_tokenizers, thief_list_configs)

def entropy_stealing(cfg, victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    BATCH_SIZE = cfg.THIEF.PER_GPU_BATCH_SIZE * max(1, n_gpu)
    set_seed(cfg, n_gpu)

    model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR, "thief-{}_victim-{}_thiefModel-{}_victimModel-{}_method-{}_epochs-{}_budget-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.ARCHITECTURE , cfg.VICTIM.ARCHITECTURE , cfg.ACTIVE.METHOD , cfg.THIEF_HYPERPARAMETERS.EPOCH, cfg.ACTIVE.BUDGET))
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    validation_accuracy_dict = {}
    global_step_dict = {}
    loss_dict = {}
    thief_model.to(device)  
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    save_thief_model(thief_model , thief_tokenizer , thief_config, model_out_dir + "/{}".format(cfg.THIEF.ARCHITECTURE))

    print("--------------- thief model setup for training ---------------")
    thief_dataset_train = thief_dataset.get_loaded_features(index = None , split = "train")
    thief_train_dataloader = DataLoader(thief_dataset_train, sampler=RandomSampler(thief_dataset_train, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    print("---------------  quering victim thief_model using thief data & created indexes ---------------")
    thief_train_features_for_victim_model = thief_dataset.get_features(split = 'train', tokenizer=victim_tokenizer, label_list=thief_dataset.get_labels())
    thief_dataset_train_for_victim = thief_dataset.get_loaded_features(index = None , split = "train" , features = thief_train_features_for_victim_model)
    thief_train_dataloader_for_victim = DataLoader(thief_dataset_train_for_victim, sampler=RandomSampler(thief_dataset_train_for_victim, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    true_train_labels_thief , entropy_train_thief , index_list_train = query_data_in_victim(cfg=cfg, theif_dataloader=thief_train_dataloader_for_victim, victim_model=victim_model, victim_tokenizer=victim_tokenizer, victim_config=victim_config)

    budget = cfg.ACTIVE.BUDGET
    unlabeled_index_list = random.sample(list(index_list_train), len(list(index_list_train)))
    labelled_index_list = unlabeled_index_list[0:1500]
    validation_index_list = unlabeled_index_list[1500:3000]
    unlabelled_index_list = list(set(unlabeled_index_list) - set(labelled_index_list) - set(validation_index_list))

    budget = budget - 3000
    cycle_num = 1
    while(budget >= 0):
        print("--------------- cycle {} budget left - {}---------------".format(cycle_num , budget))
        training_dataset = thief_dataset.get_loaded_features(index = labelled_index_list , split = "train" , true_labels = true_train_labels_thief)
        validation_dataset = thief_dataset.get_loaded_features(index = validation_index_list , split = "train" , true_labels = true_train_labels_thief)
        train_loader = DataLoader(training_dataset, sampler=RandomSampler(training_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
        validation_loader = DataLoader(validation_dataset, sampler=RandomSampler(validation_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
        global_step, tr_loss , validation_accuracy , thief_model , thief_config , thief_tokenizer = active_train(cfg , train_loader , validation_loader , thief_model , thief_config , thief_tokenizer)
        validation_accuracy_dict[cycle_num] = validation_accuracy
        global_step_dict[cycle_num] = global_step
        loss_dict[cycle_num] = tr_loss
        print("global_step = %s, average_training_loss = %s" % (global_step, tr_loss))
        print("validation accuracy", validation_accuracy)
        save_thief_model(thief_model , thief_tokenizer , thief_config, model_out_dir + "/{}".format(cfg.THIEF.ARCHITECTURE))
        save_thief_model(thief_model , thief_tokenizer , thief_config, model_out_dir + "/cycle_num-{}".format(cycle_num))

        
        # select the top 1000 data
        if budget == 0:
            break
        elif budget >= 1000:
            unlabelled_dataset = thief_dataset.get_loaded_features(index = unlabelled_index_list , split = "train" , true_labels = true_train_labels_thief)
            unlabelled_loader = DataLoader(unlabelled_dataset, sampler=RandomSampler(unlabelled_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
            index_to_entropy = get_entropy(cfg , thief_model , thief_tokenizer , thief_config , unlabelled_loader)
            unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=True)
            labelled_index_list.extend(unlabelled_index_list[:1000])
            unlabelled_index_list = unlabelled_index_list[1000:]
            budget = budget - 1000
        else:
            unlabelled_dataset = thief_dataset.get_loaded_features(index = unlabelled_index_list , split = "train" , true_labels = true_train_labels_thief)
            unlabelled_loader = DataLoader(unlabelled_dataset, sampler=RandomSampler(unlabelled_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
            index_to_entropy = get_entropy(cfg , thief_model , thief_tokenizer , thief_config , unlabelled_loader)
            unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=True)
            if budget >= len(unlabelled_index_list) & budget < 1000:
                labelled_index_list.extend(unlabelled_index_list[:len(unlabelled_index_list)])  
                unlabelled_index_list = unlabelled_index_list[len(unlabelled_index_list):]
            else:
                labelled_index_list.extend(unlabelled_index_list[:budget])
                unlabelled_index_list = unlabelled_index_list[budget:]
            budget = -1

        print("------------------- budget left - ",budget, "-------------------")
        cycle_num = cycle_num + 1
    
    with open(os.path.join(model_out_dir, "validation_results.txt"), "w") as f:
        f.write("cycle_num\tglobal_step\ttraining_loss\tvalidation_accuracy\n")
        for cycle_num in validation_accuracy_dict.keys():
            f.write("%s\t%s\t%s\t%s\t" % (cycle_num, global_step_dict[cycle_num], loss_dict[cycle_num], validation_accuracy_dict[cycle_num]))
    print(validation_accuracy_dict)

    print("--------------- thief model setup for testing ---------------")
    victim_dataset_test_features_for_thief_model = victim_dataset.get_features(split = 'test', tokenizer=thief_tokenizer, label_list=victim_dataset.get_labels())
    victim_dataset_test_features_for_victim_model = victim_dataset.get_features(split = 'test', tokenizer=victim_tokenizer, label_list=victim_dataset.get_labels())
    victim_dataset_test_for_victim = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset_test_features_for_victim_model)
    victim_dataset_test_for_thief = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset_test_features_for_thief_model)
    victim_test_dataloader_for_victim = DataLoader(victim_dataset_test_for_victim, sampler=SequentialSampler(victim_dataset_test_for_victim), batch_size=BATCH_SIZE)
    victim_test_dataloader_for_thief = DataLoader(victim_dataset_test_for_thief, sampler=SequentialSampler(victim_dataset_test_for_thief), batch_size=BATCH_SIZE)
    result_victim, preds_victim = evaluate(cfg, victim_test_dataloader_for_victim, victim_model, victim_tokenizer, victim_config)

    accuracy_list = {}
    agreement_score_list = {}
    model_folders = os.listdir(model_out_dir)
    model_folders = [x for x in model_folders if ".txt" not in x]
    model_folders.sort()
    model_folders = [os.path.join(model_out_dir, x) for x in model_folders]
    print(model_folders)
    for model_folder in model_folders:
        print("Evaluating thief_model:", model_folder)
        eval_model = AutoModelForSequenceClassification.from_pretrained(model_folder)
        eval_tokenizer = AutoTokenizer.from_pretrained(model_folder)
        eval_config = AutoConfig.from_pretrained(model_folder)
        # eval_model.to(device)
        result_thief, preds_thief = evaluate(cfg, victim_test_dataloader_for_thief, eval_model, eval_tokenizer, eval_config)
        agreement_score_list[model_folder] = agreement_score(preds_thief, preds_victim)
        accuracy_list[model_folder] = result_thief 
    print("accuracy list", accuracy_list)
    with open(os.path.join(model_out_dir, "accuracy_list.txt"), "w") as f:
        f.write("model\tagreement_score\taccuracy\n")
        for key in accuracy_list.keys():
            f.write("{}\t{}\t{}".format(key, agreement_score_list[key], accuracy_list[key]))
            f.write("\n")

def all_data_stealing(cfg , victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    BATCH_SIZE = cfg.THIEF.PER_GPU_BATCH_SIZE * max(1, n_gpu)
    set_seed(cfg, n_gpu)
    
    model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR, "thief-{}_victim-{}_thiefModel-{}_victimModel-{}_method-{}_epochs-{}_budget-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.ARCHITECTURE , cfg.VICTIM.ARCHITECTURE , cfg.ACTIVE.METHOD , cfg.THIEF_HYPERPARAMETERS.EPOCH, cfg.ACTIVE.BUDGET))
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    thief_model.to(device)  
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    
    print("--------------- thief model setup for training ---------------")
    thief_dataset_train_all = thief_dataset.get_loaded_features(index = None , split = "train")
    thief_train_dataloader_all = DataLoader(thief_dataset_train_all, sampler=RandomSampler(thief_dataset_train_all, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)

    thief_train_features_for_victim_model = thief_dataset.get_features(split = 'train', tokenizer=victim_tokenizer, label_list=thief_dataset.get_labels())
    thief_dataset_train_for_victim = thief_dataset.get_loaded_features(index = None , split = "train" , features = thief_train_features_for_victim_model)
    thief_train_dataloader_for_victim = DataLoader(thief_dataset_train_for_victim, sampler=RandomSampler(thief_dataset_train_for_victim, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    true_train_labels_thief , entropy_train_thief , index_list_train = query_data_in_victim(cfg=cfg, theif_dataloader=thief_train_dataloader_for_victim, victim_model=victim_model, victim_tokenizer=victim_tokenizer, victim_config=victim_config)

    np.random.shuffle(index_list_train)
    train_indexes = index_list_train[:int(0.8 * len(index_list_train))]
    validation_indexes = index_list_train[int(0.8 * len(index_list_train)):]

    thief_dataset_train = thief_dataset.get_loaded_features(index = train_indexes , split = "train", true_labels = true_train_labels_thief)
    thief_dataset_validation = thief_dataset.get_loaded_features(index = validation_indexes , split = "train", true_labels = true_train_labels_thief)
    thief_train_dataloader = DataLoader(thief_dataset_train, sampler=RandomSampler(thief_dataset_train, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    thief_validation_dataloader = DataLoader(thief_dataset_validation, sampler=RandomSampler(thief_dataset_validation, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)

    global_step, tr_loss , validation_accuracy , thief_model , thief_config , thief_tokenizer = active_train(cfg , thief_train_dataloader , thief_validation_dataloader , thief_model , thief_config , thief_tokenizer)
    save_thief_model(thief_model , thief_tokenizer , thief_config, model_out_dir + "/{}".format(cfg.THIEF.ARCHITECTURE))

    with open(os.path.join(model_out_dir, "validation_results.txt"), "w") as f:
        f.write("cycle_num\tglobal_step\ttraining_loss\tvalidation_accuracy\n")
        f.write("%s\t%s\t%s\t%s\t" % (1, global_step, tr_loss, validation_accuracy))

    print("global_step = %s, average_training_loss = %s , validation_accuracy = %s" % (global_step, tr_loss , validation_accuracy))

    print("--------------- thief model setup for testing ---------------")
    victim_dataset_test_features_for_thief_model = victim_dataset.get_features(split = 'test', tokenizer=thief_tokenizer, label_list=victim_dataset.get_labels())
    victim_dataset_test_features_for_victim_model = victim_dataset.get_features(split = 'test', tokenizer=victim_tokenizer, label_list=victim_dataset.get_labels())
    victim_dataset_test_for_victim = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset_test_features_for_victim_model)
    victim_dataset_test_for_thief = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset_test_features_for_thief_model)
    victim_test_dataloader_for_victim = DataLoader(victim_dataset_test_for_victim, sampler=SequentialSampler(victim_dataset_test_for_victim), batch_size=BATCH_SIZE)
    victim_test_dataloader_for_thief = DataLoader(victim_dataset_test_for_thief, sampler=SequentialSampler(victim_dataset_test_for_thief), batch_size=BATCH_SIZE)
    result_victim, preds_victim = evaluate(cfg, victim_test_dataloader_for_victim, victim_model, victim_tokenizer, victim_config)

    accuracy_list = {}
    agreement_score_list = {}
    model_folders = os.listdir(model_out_dir)
    model_folders = [x for x in model_folders if ".txt" not in x]
    model_folders.sort()
    model_folders = [os.path.join(model_out_dir, x) for x in model_folders]
    print(model_folders)
    for model_folder in model_folders:
        print("Evaluating thief_model:", model_folder)
        eval_model = AutoModelForSequenceClassification.from_pretrained(model_folder)
        eval_tokenizer = AutoTokenizer.from_pretrained(model_folder)
        eval_config = AutoConfig.from_pretrained(model_folder)
        # eval_model.to(device)
        result_thief, preds_thief = evaluate(cfg, victim_test_dataloader, eval_model, eval_tokenizer, eval_config)
        agreement_score_list[model_folder] = agreement_score(preds_thief, preds_victim)
        accuracy_list[model_folder] = result 
    print("accuracy list", accuracy_list)
    with open(os.path.join(model_out_dir, "accuracy_list.txt"), "w") as f:
        f.write("model\tagreement_score\taccuracy\n")
        for key in accuracy_list.keys():
            f.write("{}\t{}\t{}".format(key, agreement_score_list[key], accuracy_list[key]))
            f.write("\n")
   
def qbc(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, list_models : list, list_tokenizers : list , list_configs : list):
    number_of_models = len(list_models)
    validation_accuracy_dict_all_models = {}
    global_step_dict_all_models = {}
    loss_dict_all_models = {}
    for key in list_models.keys():
        validation_accuracy_dict_all_models[key] = []
        global_step_dict_all_models[key] = []
        loss_dict_all_models[key] = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    # BATCH_SIZE = cfg.THIEF_HYPERPARAMETERS.PER_GPU_BATCH_SIZE * max(1, n_gpu)
    set_seed(cfg, n_gpu)
    model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR, "thief-{}_victim-{}_thiefModel-{}_victimModel-{}_method-{}_epochs-{}_budget-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.ARCHITECTURE , cfg.VICTIM.ARCHITECTURE , cfg.ACTIVE.METHOD , cfg.THIEF_HYPERPARAMETERS.EPOCH, cfg.ACTIVE.BUDGET))
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    # else:
    #     # delete the folder and create a new one
    #     import shutil
    #     shutil.rmtree(model_out_dir)
    #     os.makedirs(model_out_dir)

    # thief_train_features = thief_dataset.get_features(split = 'train', tokenizer=victim_tokenizer, label_list=thief_dataset.get_labels())
    # thief_dataset_train_all = thief_dataset.get_loaded_features(index = None , split = "train", features = thief_train_features)
    # thief_train_dataloader_all = DataLoader(thief_dataset_train_all, sampler=RandomSampler(thief_dataset_train_all, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    # true_train_labels_thief , entropy_train_thief , index_list_train = query_data_in_victim(cfg=cfg, theif_dataloader=thief_train_dataloader_all, victim_model=victim_model, victim_tokenizer=victim_tokenizer, victim_config=victim_config)
    # thief_train_features_per_model = {}
    # for key , tokenizer in list_tokenizers.items():
    #     thief_train_features_per_model[key] = thief_dataset.get_features(split = 'train', tokenizer=tokenizer, label_list=thief_dataset.get_labels())
    
    # index_to_label_prob = {}
    # budget = cfg.ACTIVE.BUDGET

    # unlabeled_index_list = random.sample(list(index_list_train), len(list(index_list_train)))
    # labelled_index_list = unlabeled_index_list[0:1500]
    # validation_index_list = unlabeled_index_list[1500:3000]
    # unlabelled_index_list = list(set(unlabeled_index_list) - set(labelled_index_list) - set(validation_index_list))
    # budget = budget - 3000
    # cycle_num = 1

    # while(budget >= 0):
    #     for key in list_models.keys():
    #         sub_model_out_dir = os.path.join(model_out_dir, key)
    #         print("---------------model - {} cycle {} budget left - {}---------------".format(key, cycle_num , budget))
    #         training_dataset = thief_dataset.get_loaded_features(index = labelled_index_list , split = "train" , true_labels = true_train_labels_thief, features = thief_train_features_per_model[key])
    #         validation_dataset = thief_dataset.get_loaded_features(index = validation_index_list , split = "train" , true_labels = true_train_labels_thief , features = thief_train_features_per_model[key])
    #         train_loader = DataLoader(training_dataset, sampler=RandomSampler(training_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    #         validation_loader = DataLoader(validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=BATCH_SIZE)
    #         global_step, tr_loss , validation_accuracy , model , config , tokenizer = active_train(cfg , train_loader , validation_loader , list_models[key] , list_configs[key] , list_tokenizers[key])
    #         validation_accuracy_dict_all_models[key].append({cycle_num : validation_accuracy})
    #         global_step_dict_all_models[key].append({cycle_num : global_step})
    #         loss_dict_all_models[key].append({cycle_num : tr_loss})
    #         print("global_step = %s, average_training_loss = %s , validation_accuracy = %s" % (global_step, tr_loss , validation_accuracy))
    #         save_thief_model(model , tokenizer , config, sub_model_out_dir + "/cycle_num-{}".format(cycle_num))
    #         if budget > 0:
    #             unlabelled_dataset = thief_dataset.get_loaded_features(index = unlabelled_index_list , split = "train" , true_labels = true_train_labels_thief , features = thief_train_features_per_model[key])
    #             unlabelled_loader = DataLoader(unlabelled_dataset, sampler=SequentialSampler(unlabelled_dataset), batch_size=BATCH_SIZE)
    #             index_to_label_prob[key] =  get_label_probs(cfg , model , tokenizer , config , unlabelled_loader)
    
    #     unlabelled_index_list = get_index_by_vote(index_to_label_prob)
    #     if budget == 0:
    #         break
    #     elif budget >= 1000:
    #         labelled_index_list.extend(unlabelled_index_list[:1000])
    #         unlabelled_index_list = unlabelled_index_list[1000:]
    #         budget = budget - 1000
    #     elif budget >= len(unlabelled_index_list) & budget < 1000:
    #         labelled_index_list.extend(unlabelled_index_list[:len(unlabelled_index_list)])  
    #         unlabelled_index_list = unlabelled_index_list[len(unlabelled_index_list):]
    #         budget = -1
    #     else:
    #         labelled_index_list.extend(unlabelled_index_list[:budget])
    #         unlabelled_index_list = unlabelled_index_list[budget:]
    #         budget = -1
    #     print("------------------- budget left - ",budget, "-------------------")
    #     cycle_num = cycle_num + 1


    # with open(os.path.join(model_out_dir, "validation_results.txt"), "w") as f:
    #     f.write("cycle_num\tglobal_step\ttraining_loss\tvalidation_accuracy\n")
    #     for key in validation_accuracy_dict_all_models.keys():
    #         for i, dicts in enumerate(validation_accuracy_dict_all_models[key]):
    #             key2 = list(dicts.keys())[0]
    #             f.write("%s\t%s\t%s\t%s\t" % (key2, global_step_dict_all_models[key][i][key2], loss_dict_all_models[key][i][key2], validation_accuracy_dict_all_models[key][i][key2]))

    # print("--------------- thief model setup for testing ---------------")
    # victim_dataset_test_features_for_victim_model = victim_dataset.get_features(split = 'test', tokenizer=victim_tokenizer, label_list=victim_dataset.get_labels())
    # victim_dataset_test_for_victim = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset_test_features_for_victim_model)
    # victim_test_dataloader_for_victim = DataLoader(victim_dataset_test_for_victim, sampler=SequentialSampler(victim_dataset_test_for_victim), batch_size=BATCH_SIZE)
    # result_victim, preds_victim = evaluate(cfg, victim_test_dataloader_for_victim, victim_model, victim_tokenizer, victim_config)
    # print("victim accuracy", result_victim)

    # thief_test_loader_per_model = {}
    # for key , tokenizer in list_tokenizers.items():
    #     victim_dataset_test_features_for_thief_model = victim_dataset.get_features(split = 'test', tokenizer=tokenizer, label_list=victim_dataset.get_labels())
    #     victim_dataset_test_for_thief = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset_test_features_for_thief_model)
    #     victim_test_dataloader_for_thief = DataLoader(victim_dataset_test_for_thief, sampler=SequentialSampler(victim_dataset_test_for_thief), batch_size=BATCH_SIZE)
    #     thief_test_loader_per_model[key] = victim_test_dataloader_for_thief


    accuracy_list = {}
    agreement_score_list = {}

    for key in list_models.keys():
        accuracy_list[key] = []
        agreement_score_list[key] = []

    for key in list_models.keys():
        print("Evaluating thief_model:", key)
        sub_model_out_dir = os.path.join(model_out_dir, key)
        list_models = os.listdir(sub_model_out_dir)
        list_models = [x for x in list_models if ".txt" not in x]
        list_models.sort()
        list_models = [os.path.join(sub_model_out_dir, x) for x in list_models]
        print(list_models)
        for model_folder in list_models:
            # eval_model = AutoModelForSequenceClassification.from_pretrained(model_folder, num_labels=cfg.VICTIM.NUM_LABELS)
            # eval_tokenizer = AutoTokenizer.from_pretrained(model_folder)
            # eval_config = AutoConfig.from_pretrained(model_folder)
            result_thief = 0.1
            # result_thief, preds_thief = evaluate(cfg, thief_test_loader_per_model[key], eval_model, eval_tokenizer, eval_config)
            # agreement_score_list[key] = agreement_score(preds_thief, preds_victim)
            accuracy_list[key].append({model_folder : result_thief})
    print("accuracy list", accuracy_list)
    with open(os.path.join(model_out_dir, "accuracy_list.txt"), "w") as f:
        f.write("model\tagreement_score\taccuracy\n")
        for key in accuracy_list.keys():
            for i, dicts in enumerate(accuracy_list[key]):
                key2 = dicts.keys()
                key2 = list(key2)
                key2 = key2[0]
                f.write("{}\t{}\t{}".format(key2, accuracy_list[key][i][key2], accuracy_list[key][i][key2]))
                f.write("\n")

def semi_supervised_entropy(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, thief_model, thief_tokenizer, thief_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    BATCH_SIZE = cfg.THIEF.PER_GPU_BATCH_SIZE * max(1, n_gpu)
    set_seed(cfg, n_gpu)

    model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR, "thief-{}_victim-{}_thiefModel-{}_victimModel-{}_method-{}_epochs-{}_budget-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.ARCHITECTURE , cfg.VICTIM.ARCHITECTURE , cfg.ACTIVE.METHOD , cfg.THIEF_HYPERPARAMETERS.EPOCH, cfg.ACTIVE.BUDGET))
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    validation_accuracy_dict = {}
    global_step_dict = {}
    loss_dict = {}
    thief_model.to(device)  
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    save_thief_model(thief_model , thief_tokenizer , thief_config, model_out_dir + "/{}".format(cfg.THIEF.ARCHITECTURE))

    print("--------------- thief model setup for training ---------------")
    thief_dataset_train = thief_dataset.get_loaded_features(index = None , split = "train")
    thief_train_dataloader = DataLoader(thief_dataset_train, sampler=RandomSampler(thief_dataset_train, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    print("---------------  quering victim thief_model using thief data & created indexes ---------------")
    thief_train_features_for_victim_model = thief_dataset.get_features(split = 'train', tokenizer=victim_tokenizer, label_list=thief_dataset.get_labels())
    thief_dataset_train_for_victim = thief_dataset.get_loaded_features(index = None , split = "train" , features = thief_train_features_for_victim_model)
    thief_train_dataloader_for_victim = DataLoader(thief_dataset_train_for_victim, sampler=RandomSampler(thief_dataset_train_for_victim, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    true_train_labels_thief , entropy_train_thief , index_list_train = query_data_in_victim(cfg=cfg, theif_dataloader=thief_train_dataloader_for_victim, victim_model=victim_model, victim_tokenizer=victim_tokenizer, victim_config=victim_config)

    budget = cfg.ACTIVE.BUDGET
    unlabeled_index_list = random.sample(list(index_list_train), len(list(index_list_train)))
    labelled_index_list = unlabeled_index_list[0:1000]
    validation_index_list = unlabeled_index_list[1000:2000]
    unlabelled_index_list = list(set(unlabeled_index_list) - set(labelled_index_list) - set(validation_index_list))
    budget = budget - 2000

    thief_model_cycle = None
    thief_tokenizer_cycle = None
    thief_config_cycle = None

    cycle_num = 1
    while(budget >= 0):
        print("--------------- cycle {} budget left - {}---------------".format(cycle_num , budget))
        thief_model_cycle = copy.deepcopy(thief_model)
        thief_tokenizer_cycle = copy.deepcopy(thief_tokenizer)
        thief_config_cycle = copy.deepcopy(thief_config)
        training_dataset = thief_dataset.get_loaded_features(index = labelled_index_list , split = "train" , true_labels = true_train_labels_thief)
        validation_dataset = thief_dataset.get_loaded_features(index = validation_index_list , split = "train" , true_labels = true_train_labels_thief)
        train_loader = DataLoader(training_dataset, sampler=RandomSampler(training_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
        validation_loader = DataLoader(validation_dataset, sampler=RandomSampler(validation_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
        global_step, tr_loss , validation_accuracy , thief_model , thief_config , thief_tokenizer = active_train(cfg , train_loader , validation_loader , thief_model_cycle , thief_config_cycle , thief_tokenizer_cycle)
        validation_accuracy_dict[cycle_num] = validation_accuracy
        global_step_dict[cycle_num] = global_step
        loss_dict[cycle_num] = tr_loss
        print("global_step = %s, average_training_loss = %s" % (global_step, tr_loss))
        print("validation accuracy", validation_accuracy)
        save_thief_model(thief_model_cycle , thief_tokenizer_cycle , thief_config_cycle, model_out_dir + "/{}".format(cfg.THIEF.ARCHITECTURE))
        save_thief_model(thief_model_cycle , thief_tokenizer_cycle , thief_config_cycle, model_out_dir + "/cycle_num-{}".format(cycle_num))

        
        # select the top 1000 data
        if budget == 0:
            break
        elif budget >= 1000:
            unlabelled_dataset = thief_dataset.get_loaded_features(index = unlabelled_index_list , split = "train" , true_labels = true_train_labels_thief)
            unlabelled_loader = DataLoader(unlabelled_dataset, sampler=RandomSampler(unlabelled_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
            index_to_entropy = get_entropy(cfg , thief_model_cycle , thief_tokenizer_cycle , thief_config_cycle , unlabelled_loader)
            unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=True)
            labelled_index_list.extend(unlabelled_index_list[:1000])
            unlabelled_index_list = unlabelled_index_list[1000:]
            budget = budget - 1000
        else:
            unlabelled_dataset = thief_dataset.get_loaded_features(index = unlabelled_index_list , split = "train" , true_labels = true_train_labels_thief)
            unlabelled_loader = DataLoader(unlabelled_dataset, sampler=RandomSampler(unlabelled_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
            index_to_entropy = get_entropy(cfg , thief_model_cycle , thief_tokenizer_cycle , thief_config_cycle , unlabelled_loader)
            unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=True)
            if budget >= len(unlabelled_index_list) & budget < 1000:
                labelled_index_list.extend(unlabelled_index_list[:len(unlabelled_index_list)])  
                unlabelled_index_list = unlabelled_index_list[len(unlabelled_index_list):]
            else:
                labelled_index_list.extend(unlabelled_index_list[:budget])
                unlabelled_index_list = unlabelled_index_list[budget:]
            budget = -1

        print("------------------- budget left - ",budget, "-------------------")
        cycle_num = cycle_num + 1

    # find_entropy of remaining data
    index_to_entropy = get_entropy(cfg , thief_model_cycle , thief_tokenizer_cycle , thief_config_cycle , unlabelled_loader)
    # choose top 3000 data with high confidence
    unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=False)
    labelled_index_list.extend(unlabelled_index_list[:3000])

    # train the model on all the data
    thief_model_cycle = copy.deepcopy(thief_model)
    thief_tokenizer_cycle = copy.deepcopy(thief_tokenizer)
    thief_config_cycle = copy.deepcopy(thief_config)
    training_dataset = thief_dataset.get_loaded_features(index = labelled_index_list , split = "train" , true_labels = true_train_labels_thief)
    validation_dataset = thief_dataset.get_loaded_features(index = validation_index_list , split = "train" , true_labels = true_train_labels_thief)
    train_loader = DataLoader(training_dataset, sampler=RandomSampler(training_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    validation_loader = DataLoader(validation_dataset, sampler=RandomSampler(validation_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    global_step, tr_loss , validation_accuracy , thief_model , thief_config , thief_tokenizer = active_train(cfg , train_loader , validation_loader , thief_model_cycle , thief_config_cycle , thief_tokenizer_cycle)
    validation_accuracy_dict["semi_supervised"] = validation_accuracy
    global_step_dict["semi_supervised"] = global_step
    loss_dict["semi_supervised"] = tr_loss
    print("global_step = %s, average_training_loss = %s validation_accuracy = %s" % (global_step, tr_loss , validation_accuracy))
    
    with open(os.path.join(model_out_dir, "validation_results.txt"), "w") as f:
        f.write("cycle_num\tglobal_step\ttraining_loss\tvalidation_accuracy\n")
        for cycle_num in validation_accuracy_dict.keys():
            f.write("%s\t%s\t%s\t%s\t" % (cycle_num, global_step_dict[cycle_num], loss_dict[cycle_num], validation_accuracy_dict[cycle_num]))
    print(validation_accuracy_dict)

    print("--------------- thief model setup for testing ---------------")
    victim_dataset_test_features_for_thief_model = victim_dataset.get_features(split = 'test', tokenizer=thief_tokenizer, label_list=victim_dataset.get_labels())
    victim_dataset_test_features_for_victim_model = victim_dataset.get_features(split = 'test', tokenizer=victim_tokenizer, label_list=victim_dataset.get_labels())
    victim_dataset_test_for_victim = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset_test_features_for_victim_model)
    victim_dataset_test_for_thief = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset_test_features_for_thief_model)
    victim_test_dataloader_for_victim = DataLoader(victim_dataset_test_for_victim, sampler=SequentialSampler(victim_dataset_test_for_victim), batch_size=BATCH_SIZE)
    victim_test_dataloader_for_thief = DataLoader(victim_dataset_test_for_thief, sampler=SequentialSampler(victim_dataset_test_for_thief), batch_size=BATCH_SIZE)
    result_victim, preds_victim = evaluate(cfg, victim_test_dataloader_for_victim, victim_model, victim_tokenizer, victim_config)

    accuracy_list = {}
    agreement_score_list = {}
    model_folders = os.listdir(model_out_dir)
    model_folders = [x for x in model_folders if ".txt" not in x]
    model_folders.sort()
    model_folders = [os.path.join(model_out_dir, x) for x in model_folders]
    print(model_folders)
    for model_folder in model_folders:
        print("Evaluating thief_model:", model_folder)
        eval_model = AutoModelForSequenceClassification.from_pretrained(model_folder)
        eval_tokenizer = AutoTokenizer.from_pretrained(model_folder)
        eval_config = AutoConfig.from_pretrained(model_folder)
        # eval_model.to(device)
        result_thief, preds_thief = evaluate(cfg, victim_test_dataloader_for_thief, eval_model, eval_tokenizer, eval_config)
        agreement_score_list[model_folder] = agreement_score(preds_thief, preds_victim)
        accuracy_list[model_folder] = result_thief 
    print("accuracy list", accuracy_list)
    with open(os.path.join(model_out_dir, "accuracy_list.txt"), "w") as f:
        f.write("model\tagreement_score\taccuracy\n")
        for key in accuracy_list.keys():
            f.write("{}\t{}\t{}".format(key, agreement_score_list[key], accuracy_list[key]))
            f.write("\n")

def semi_supervised_qbc(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, list_models : list, list_tokenizers : list , list_configs : list):
    number_of_models = len(list_models)
    validation_accuracy_dict_all_models = {}
    global_step_dict_all_models = {}
    loss_dict_all_models = {}
    for key in list_models.keys():
        validation_accuracy_dict_all_models[key] = []
        global_step_dict_all_models[key] = []
        loss_dict_all_models[key] = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    BATCH_SIZE = cfg.THIEF.PER_GPU_BATCH_SIZE * max(1, n_gpu)
    set_seed(cfg, n_gpu)
    model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR, "thief-{}_victim-{}_thiefModel-{}_victimModel-{}_method-{}_epochs-{}_budget-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.ARCHITECTURE , cfg.VICTIM.ARCHITECTURE , cfg.ACTIVE.METHOD , cfg.THIEF_HYPERPARAMETERS.EPOCH, cfg.ACTIVE.BUDGET))
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    else:
        # delete the folder and create a new one
        import shutil
        shutil.rmtree(model_out_dir)
        os.makedirs(model_out_dir)

    thief_train_features = thief_dataset.get_features(split = 'train', tokenizer=victim_tokenizer, label_list=thief_dataset.get_labels())
    thief_dataset_train_all = thief_dataset.get_loaded_features(index = None , split = "train", features = thief_train_features)
    thief_train_dataloader_all = DataLoader(thief_dataset_train_all, sampler=RandomSampler(thief_dataset_train_all, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    true_train_labels_thief , entropy_train_thief , index_list_train = query_data_in_victim(cfg=cfg, theif_dataloader=thief_train_dataloader_all, victim_model=victim_model, victim_tokenizer=victim_tokenizer, victim_config=victim_config)
    thief_train_features_per_model = {}
    for key , tokenizer in list_tokenizers.items():
        thief_train_features_per_model[key] = thief_dataset.get_features(split = 'train', tokenizer=tokenizer, label_list=thief_dataset.get_labels())
    
    index_to_label_prob = {}
    budget = cfg.ACTIVE.BUDGET

    unlabeled_index_list = random.sample(list(index_list_train), len(list(index_list_train)))
    labelled_index_list = unlabeled_index_list[0:1500]
    validation_index_list = unlabeled_index_list[1500:3000]
    unlabelled_index_list = list(set(unlabeled_index_list) - set(labelled_index_list) - set(validation_index_list))
    budget = budget - 3000
    cycle_num = 1

    while(budget >= 0):
        for key in list_models.keys():
            sub_model_out_dir = os.path.join(model_out_dir, key)
            theif_model_cycle = copy.deepcopy(list_models[key])
            theif_tokenizer_cycle = copy.deepcopy(list_tokenizers[key])
            theif_config_cycle = copy.deepcopy(list_configs[key])
            print("---------------model - {} cycle {} budget left - {}---------------".format(key, cycle_num , budget))
            training_dataset = thief_dataset.get_loaded_features(index = labelled_index_list , split = "train" , true_labels = true_train_labels_thief, features = thief_train_features_per_model[key])
            validation_dataset = thief_dataset.get_loaded_features(index = validation_index_list , split = "train" , true_labels = true_train_labels_thief , features = thief_train_features_per_model[key])
            train_loader = DataLoader(training_dataset, sampler=RandomSampler(training_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
            validation_loader = DataLoader(validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=BATCH_SIZE)
            global_step, tr_loss , validation_accuracy , theif_model_cycle , theif_config_cycle , theif_tokenizer_cycle = active_train(cfg , train_loader , validation_loader , theif_model_cycle , theif_config_cycle , theif_tokenizer_cycle)
            validation_accuracy_dict_all_models[key].append({cycle_num : validation_accuracy})
            global_step_dict_all_models[key].append({cycle_num : global_step})
            loss_dict_all_models[key].append({cycle_num : tr_loss})
            print("global_step = %s, average_training_loss = %s , validation_accuracy = %s" % (global_step, tr_loss , validation_accuracy))
            save_thief_model(theif_model_cycle , theif_tokenizer_cycle , theif_config_cycle, sub_model_out_dir + "/cycle_num-{}".format(cycle_num))
            save_thief_model(theif_model_cycle , theif_tokenizer_cycle , theif_config_cycle, sub_model_out_dir + "/{}".format(cfg.THIEF.ARCHITECTURE))
            if budget > 0:
                unlabelled_dataset = thief_dataset.get_loaded_features(index = unlabelled_index_list , split = "train" , true_labels = true_train_labels_thief , features = thief_train_features_per_model[key])
                unlabelled_loader = DataLoader(unlabelled_dataset, sampler=SequentialSampler(unlabelled_dataset), batch_size=BATCH_SIZE)
                index_to_label_prob[key] =  get_label_probs(cfg , theif_model_cycle , theif_tokenizer_cycle , theif_config_cycle , unlabelled_loader)
    
        unlabelled_index_list = get_index_by_vote(index_to_label_prob)
        if budget == 0:
            break
        elif budget >= 1000:
            labelled_index_list.extend(unlabelled_index_list[:1000])
            unlabelled_index_list = unlabelled_index_list[1000:]
            budget = budget - 1000
        elif budget >= len(unlabelled_index_list) & budget < 1000:
            labelled_index_list.extend(unlabelled_index_list[:len(unlabelled_index_list)])  
            unlabelled_index_list = unlabelled_index_list[len(unlabelled_index_list):]
            budget = -1
        else:
            labelled_index_list.extend(unlabelled_index_list[:budget])
            unlabelled_index_list = unlabelled_index_list[budget:]
            budget = -1
        print("------------------- budget left - ",budget, "-------------------")
        cycle_num = cycle_num + 1


    # find_entropy of remaining data
    index_to_label_prob[key] =  get_label_probs(cfg , theif_model_cycle , theif_tokenizer_cycle , theif_config_cycle , unlabelled_loader)
    index_to_entropy = get_entropy(cfg , theif_model_cycle , theif_tokenizer_cycle , theif_config_cycle , unlabelled_loader)
    unlabelled_index_list = get_index_by_vote(index_to_label_prob)
    # chose last 3000 data with high confidence
    labelled_index_list.extend(unlabelled_index_list[-3000:])

    with open(os.path.join(model_out_dir, "validation_results.txt"), "w") as f:
        f.write("cycle_num\tglobal_step\ttraining_loss\tvalidation_accuracy\n")
        for key in validation_accuracy_dict_all_models.keys():
            for i, dicts in enumerate(validation_accuracy_dict_all_models[key]):
                key2 = list(dicts.keys())[0]
                f.write("%s\t%s\t%s\t%s\t" % (key2, global_step_dict_all_models[key][i][key2], loss_dict_all_models[key][i][key2], validation_accuracy_dict_all_models[key][i][key2]))

    print("--------------- thief model setup for testing ---------------")
    victim_dataset_test_features_for_victim_model = victim_dataset.get_features(split = 'test', tokenizer=victim_tokenizer, label_list=victim_dataset.get_labels())
    victim_dataset_test_for_victim = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset_test_features_for_victim_model)
    victim_test_dataloader_for_victim = DataLoader(victim_dataset_test_for_victim, sampler=SequentialSampler(victim_dataset_test_for_victim), batch_size=BATCH_SIZE)
    result_victim, preds_victim = evaluate(cfg, victim_test_dataloader_for_victim, victim_model, victim_tokenizer, victim_config)
    print("victim accuracy", result_victim)

    thief_test_loader_per_model = {}
    for key , tokenizer in list_tokenizers.items():
        victim_dataset_test_features_for_thief_model = victim_dataset.get_features(split = 'test', tokenizer=tokenizer, label_list=victim_dataset.get_labels())
        victim_dataset_test_for_thief = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset_test_features_for_thief_model)
        victim_test_dataloader_for_thief = DataLoader(victim_dataset_test_for_thief, sampler=SequentialSampler(victim_dataset_test_for_thief), batch_size=BATCH_SIZE)
        thief_test_loader_per_model[key] = victim_test_dataloader_for_thief


    accuracy_list = {}
    agreement_score_list = {}

    for key in list_models.keys():
        accuracy_list[key] = []
        agreement_score_list[key] = []

    for key in list_models.keys():
        print("Evaluating thief_model:", key)
        sub_model_out_dir = os.path.join(model_out_dir, key)
        list_models = os.listdir(sub_model_out_dir)
        list_models = [x for x in list_models if ".txt" not in x]
        list_models.sort()
        list_models = [os.path.join(sub_model_out_dir, x) for x in list_models]
        print(list_models)
        for model_folder in list_models:
            eval_model = AutoModelForSequenceClassification.from_pretrained(model_folder, num_labels=cfg.VICTIM.NUM_LABELS)
            eval_tokenizer = AutoTokenizer.from_pretrained(model_folder)
            eval_config = AutoConfig.from_pretrained(model_folder)
            result_thief, preds_thief = evaluate(cfg, thief_test_loader_per_model[key], eval_model, eval_tokenizer, eval_config)
            agreement_score_list[key] = agreement_score(preds_thief, preds_victim)
            accuracy_list[key].append({model_folder : result_thief})
    print("accuracy list", accuracy_list)
    with open(os.path.join(model_out_dir, "accuracy_list.txt"), "w") as f:
        f.write("model\tagreement_score\taccuracy\n")
        for key in accuracy_list.keys():
            for i, dicts in enumerate(accuracy_list[key]):
                key2 = list(dicts.keys())[0]
                f.write("{}\t{}\t{}".format(key2, agreement_score_list[key][i][key2], accuracy_list[key][i][key2]))
                f.write("\n")
   

