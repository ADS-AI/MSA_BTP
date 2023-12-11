import os
import json
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from ...utils.text.train_utils import query_data_in_victim, active_train , evaluate, get_entropy, set_seed, agreement_score, get_label_probs, get_index_by_vote
from ...utils.text.train_metrics import metrics
from ...utils.text.load_data_and_models import load_untrained_thief_model, save_thief_model
from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)


def simple_semi_supervised_entropy_stealing(cfg, victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    BATCH_SIZE = cfg.THIEF_HYPERPARAMETERS.PER_GPU_BATCH_SIZE * max(1, n_gpu)
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

    print("--------------- victim features for querrimng victim model ---------------")
    thief_train_features_for_victim_model = thief_dataset.get_features(split = 'train', tokenizer=victim_tokenizer, label_list=thief_dataset.get_labels())
   
    # true victim info 
    true_train_labels_thief = {}
    entropy_train_thief = {}
    index_list_train = []

    # select the top 2000 examples to query in victim model
    budget = cfg.ACTIVE.BUDGET
    unlabeled_index_list = random.sample(list(range(len(thief_train_features_for_victim_model))), len(list(thief_train_features_for_victim_model)))
    labelled_index_list = unlabeled_index_list[0:1000]
    validation_index_list = unlabeled_index_list[1000:2000]
    unlabelled_index_list = list(set(unlabeled_index_list) - set(labelled_index_list) - set(validation_index_list))
    index_to_query = labelled_index_list + validation_index_list

    print("--------------- querying victim model ---------------")
    thief_dataset_train_for_victim = thief_dataset.get_loaded_features(index = index_to_query , split = "train" , features = thief_train_features_for_victim_model)
    thief_train_dataloader_for_victim = DataLoader(thief_dataset_train_for_victim, sampler=SequentialSampler(thief_dataset_train_for_victim), batch_size=BATCH_SIZE)
    true_train_labels_thief_sub , entropy_train_thief_sub , index_list_train_sub = query_data_in_victim(cfg=cfg, theif_dataloader=thief_train_dataloader_for_victim, victim_model=victim_model, victim_tokenizer=victim_tokenizer, victim_config=victim_config)
    true_train_labels_thief.update(true_train_labels_thief_sub)
    entropy_train_thief.update(entropy_train_thief_sub)
    index_list_train.extend(list(index_list_train_sub))

    total_cycles = cfg.ACTIVE.CYCLES
    budget_used = 2000
    budget = budget - 2000
    budget_per_cycle = budget / total_cycles
    budget_last_cycle = budget - (total_cycles - 1) * budget_per_cycle

    for cycle in range(1 , total_cycles + 1):
        print("--------------- cycle {} budget used - {}---------------".format(cycle , budget_used))
        training_dataset = thief_dataset.get_loaded_features(index = labelled_index_list , split = "train" , true_labels = true_train_labels_thief)
        validation_dataset = thief_dataset.get_loaded_features(index = validation_index_list , split = "train" , true_labels = true_train_labels_thief)
        train_loader = DataLoader(training_dataset, sampler=RandomSampler(training_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
        validation_loader = DataLoader(validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=BATCH_SIZE)
        global_step, tr_loss , validation_accuracy , thief_model , thief_config , thief_tokenizer = active_train(cfg , train_loader , validation_loader , thief_model , thief_config , thief_tokenizer)
        validation_accuracy_dict[cycle] = validation_accuracy
        global_step_dict[cycle] = global_step
        loss_dict[cycle] = tr_loss
        print("global_step = %s, average_training_loss = %s" % (global_step, tr_loss))
        print("validation accuracy", validation_accuracy)
        save_thief_model(thief_model , thief_tokenizer , thief_config, model_out_dir + "/{}".format(cfg.THIEF.ARCHITECTURE))
        save_thief_model(thief_model , thief_tokenizer , thief_config, model_out_dir + "/cycle-{}".format(cycle))

        unlabelled_dataset = thief_dataset.get_loaded_features(index = unlabelled_index_list , split = "train" , true_labels = true_train_labels_thief)
        unlabelled_loader = DataLoader(unlabelled_dataset, sampler=RandomSampler(unlabelled_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
        index_to_entropy = get_entropy(cfg , thief_model , thief_tokenizer , thief_config , unlabelled_loader)
        unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=True)
        if cycle == total_cycles:
            indexes_to_query = unlabelled_index_list[:budget_last_cycle]
            unlabelled_index_list = unlabelled_index_list[budget_per_cycle:]
            budget_used = budget_used + budget_last_cycle
            budget = budget - budget_last_cycle
        else:
            indexes_to_query = unlabelled_index_list[:budget_per_cycle]
            unlabelled_index_list = unlabelled_index_list[budget_per_cycle:]
            budget_used = budget_used + budget_per_cycle
            budget = budget - budget_per_cycle
        labelled_index_list.extend(indexes_to_query)

        print("--------------- querying victim model ---------------")
        thief_dataset_train_for_victim = thief_dataset.get_loaded_features(index = index_to_query , split = "train" , features = thief_train_features_for_victim_model)
        thief_train_dataloader_for_victim = DataLoader(thief_dataset_train_for_victim, sampler=SequentialSampler(thief_dataset_train_for_victim), batch_size=BATCH_SIZE)
        true_train_labels_thief_sub , entropy_train_thief_sub , index_list_train_sub = query_data_in_victim(cfg=cfg, theif_dataloader=thief_train_dataloader_for_victim, victim_model=victim_model, victim_tokenizer=victim_tokenizer, victim_config=victim_config)
        true_train_labels_thief.update(true_train_labels_thief_sub)
        entropy_train_thief.update(entropy_train_thief_sub)
        index_list_train.extend(list(index_list_train_sub))

        
    # with open(os.path.join(model_out_dir, "validation_results.txt"), "w") as f:
    #     f.write("cycle\tglobal_step\ttraining_loss\tvalidation_accuracy\n")
    #     for cycle in validation_accuracy_dict.keys():
    #         f.write("%s\t%s\t%s\t%s\t" % (cycle, global_step_dict[cycle], loss_dict[cycle], validation_accuracy_dict[cycle]))
    print(validation_accuracy_dict)

    # doing semi supervised learning
    index_to_entropy = get_entropy(cfg , thief_model , thief_tokenizer, thief_config , unlabelled_loader)
    unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=False)
    # select top 1000 examples with least entropy
    index_to_query = unlabelled_index_list[:1000]
    labelled_index_list.extend(unlabelled_index_list[:1000])
    thief_dataset_train_for_thief = thief_dataset.get_loaded_features(index = index_to_query , split = "train" , true_labels = true_train_labels_thief)
    thief_train_dataloader_for_thief = DataLoader(thief_dataset_train_for_thief, sampler=SequentialSampler(thief_dataset_train_for_thief), batch_size=BATCH_SIZE)
    true_train_labels_thief_sub , entropy_train_thief_sub , index_list_train_sub = query_data_in_victim(cfg=cfg, theif_dataloader=thief_train_dataloader_for_thief, victim_model=thief_model, victim_tokenizer=thief_tokenizer, victim_config=thief_config)
    true_train_labels_thief.update(true_train_labels_thief_sub)
    entropy_train_thief.update(entropy_train_thief_sub)
    index_list_train.extend(list(index_list_train_sub))

    training_dataset = thief_dataset.get_loaded_features(index = labelled_index_list , split = "train" , true_labels = true_train_labels_thief)
    validation_dataset = thief_dataset.get_loaded_features(index = validation_index_list , split = "train" , true_labels = true_train_labels_thief)
    train_loader = DataLoader(training_dataset, sampler=RandomSampler(training_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    validation_loader = DataLoader(validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=BATCH_SIZE)
    global_step, tr_loss , validation_accuracy , thief_model , thief_config , thief_tokenizer = active_train(cfg , train_loader , validation_loader , thief_model , thief_config , thief_tokenizer)
    validation_accuracy_dict["semi_supervised"] = validation_accuracy
    global_step_dict["semi_supervised"] = global_step
    loss_dict["semi_supervised"] = tr_loss
    print("semisupervised global_step = %s, average_training_loss = %s validation_accuracy = %s" % (global_step, tr_loss , validation_accuracy))



    print("--------------- thief model setup for testing ---------------")
    victim_dataset_test_for_victim = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset.get_features(split = 'test', tokenizer=victim_tokenizer, label_list=victim_dataset.get_labels()))
    victim_dataset_test_for_thief = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset.get_features(split = 'test', tokenizer=thief_tokenizer, label_list=victim_dataset.get_labels()))
    victim_test_dataloader_for_victim = DataLoader(victim_dataset_test_for_victim, sampler=SequentialSampler(victim_dataset_test_for_victim), batch_size=BATCH_SIZE)
    victim_test_dataloader_for_thief = DataLoader(victim_dataset_test_for_thief, sampler=SequentialSampler(victim_dataset_test_for_thief), batch_size=BATCH_SIZE)
    
    # Evaluate victim model on victim test data
    result_victim = evaluate(cfg, victim_test_dataloader_for_victim, victim_model, victim_tokenizer, victim_config)

    metrics_list = {}
    model_folders = os.listdir(model_out_dir)
    model_folders = [x for x in model_folders if ".txt" not in x]
    model_folders.sort()
    model_folders = [os.path.join(model_out_dir, x) for x in model_folders]
    model_folders_paths =  [os.path.join(model_out_dir, x) for x in model_folders]

    print(model_folders)
    for i , model_folder in enumerate(model_folders_paths):
        print("Evaluating thief_model:", model_folder)
        eval_model = AutoModelForSequenceClassification.from_pretrained(model_folder)
        eval_tokenizer = AutoTokenizer.from_pretrained(model_folder)
        eval_config = AutoConfig.from_pretrained(model_folder)

        # Evaluate thief model on victim test data
        result_thief, preds_thief = evaluate(cfg, victim_test_dataloader_for_thief, eval_model, eval_tokenizer, eval_config)
        metrics_list[model_folders[i]] = metrics(result_thief['true_labels'],  result_thief['preds'] , result_victim['preds'])

    print("metrics - ", metrics_list)
    
    with open(os.path.join(model_out_dir, "metrics_logs.json"), "w") as f:
        json.dump(metrics_list, f)

def simple_semi_supervised_qbc_stealing(cfg, victim_dataset, thief_dataset, victim_model, victim_tokenizer, victim_config, list_models : list, list_tokenizers : list , list_configs : list):
    pass
    # number_of_models = len(list_models)
    # validation_accuracy_dict_all_models = {}
    # global_step_dict_all_models = {}
    # loss_dict_all_models = {}
    # for key in list_models.keys():
    #     validation_accuracy_dict_all_models[key] = []
    #     global_step_dict_all_models[key] = []
    #     loss_dict_all_models[key] = []
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # n_gpu = torch.cuda.device_count()
    # BATCH_SIZE = cfg.THIEF_HYPERPARAMETERS.PER_GPU_BATCH_SIZE * max(1, n_gpu)
    # set_seed(cfg, n_gpu)
    # model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR, "thief-{}_victim-{}_thiefModel-{}_victimModel-{}_method-{}_epochs-{}_budget-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.ARCHITECTURE , cfg.VICTIM.ARCHITECTURE , cfg.ACTIVE.METHOD , cfg.THIEF_HYPERPARAMETERS.EPOCH, cfg.ACTIVE.BUDGET))
    # if not os.path.exists(model_out_dir):
    #     os.makedirs(model_out_dir)


    # # victim features for querrimng victim model
    # thief_train_features_for_victim_model = thief_dataset.get_features(split = 'train', tokenizer=victim_tokenizer, label_list=thief_dataset.get_labels())

    # # true victim info 
    # true_train_labels_thief = {}
    # entropy_train_thief = {}
    # index_list_train = []

    # # select the top 2000 examples to query in victim model
    # budget = cfg.ACTIVE.BUDGET
    # unlabeled_index_list = random.sample(list(range(len(thief_train_features_for_victim_model))), len(list(thief_train_features_for_victim_model)))
    # labelled_index_list = unlabeled_index_list[0:1000]
    # validation_index_list = unlabeled_index_list[1000:2000]
    # unlabelled_index_list = list(set(unlabeled_index_list) - set(labelled_index_list) - set(validation_index_list))
    # index_to_query = labelled_index_list + validation_index_list

    # print("--------------- querying victim model ---------------")
    # thief_dataset_train_for_victim = thief_dataset.get_loaded_features(index = index_to_query , split = "train" , features = thief_train_features_for_victim_model)
    # thief_train_dataloader_for_victim = DataLoader(thief_dataset_train_for_victim, sampler=SequentialSampler(thief_dataset_train_for_victim), batch_size=BATCH_SIZE)
    # true_train_labels_thief_sub , entropy_train_thief_sub , index_list_train_sub = query_data_in_victim(cfg=cfg, theif_dataloader=thief_train_dataloader_for_victim, victim_model=victim_model, victim_tokenizer=victim_tokenizer, victim_config=victim_config)
    # true_train_labels_thief.update(true_train_labels_thief_sub)
    # entropy_train_thief.update(entropy_train_thief_sub)
    # index_list_train.extend(list(index_list_train_sub))


    # thief_train_features_per_model = {}
    # for key , tokenizer in list_tokenizers.items():
    #     thief_train_features_per_model[key] = thief_dataset.get_features(split = 'train', tokenizer=tokenizer, label_list=thief_dataset.get_labels())
    
    # index_to_label_prob = {}
    # budget = cfg.ACTIVE.BUDGET
    # total_cycles = cfg.ACTIVE.CYCLES
    # budget_used = 2000
    # budget = budget - 2000
    # budget_per_cycle = budget / total_cycles
    # budget_last_cycle = budget - (total_cycles - 1) * budget_per_cycle


    # for cycle in range(1 , total_cycles + 1):
    #     for key in list_models.keys():
    #         sub_model_out_dir = os.path.join(model_out_dir, key)
    #         print("--------------- cycle {} budget used - {}---------------".format(cycle , budget_used))
    #         training_dataset = thief_dataset.get_loaded_features(index = labelled_index_list , split = "train" , true_labels = true_train_labels_thief, features = thief_train_features_per_model[key])
    #         validation_dataset = thief_dataset.get_loaded_features(index = validation_index_list , split = "train" , true_labels = true_train_labels_thief , features = thief_train_features_per_model[key])
    #         train_loader = DataLoader(training_dataset, sampler=RandomSampler(training_dataset, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    #         validation_loader = DataLoader(validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=BATCH_SIZE)
    #         global_step, tr_loss , validation_accuracy , model , config , tokenizer = active_train(cfg , train_loader , validation_loader , list_models[key] , list_configs[key] , list_tokenizers[key])
    #         validation_accuracy_dict_all_models[key].append({cycle : validation_accuracy})
    #         global_step_dict_all_models[key].append({cycle : global_step})
    #         loss_dict_all_models[key].append({cycle : tr_loss})
    #         print("model= %s, global_step = %s, average_training_loss = %s , validation_accuracy = %s" % (key, global_step, tr_loss , validation_accuracy))
    #         save_thief_model(model , tokenizer , config, sub_model_out_dir + "/cycle-{}".format(cycle))
    #         save_thief_model(model , tokenizer , config, sub_model_out_dir + "/{}".format(cfg.THIEF.ARCHITECTURE))

    #         unlabelled_dataset = thief_dataset.get_loaded_features(index = unlabelled_index_list , split = "train" , true_labels = true_train_labels_thief , features = thief_train_features_per_model[key])
    #         unlabelled_loader = DataLoader(unlabelled_dataset, sampler=SequentialSampler(unlabelled_dataset), batch_size=BATCH_SIZE)
    #         index_to_label_prob[key] =  get_label_probs(cfg , model , tokenizer , config , unlabelled_loader)
    #         unlabelled_index_list = get_index_by_vote(index_to_label_prob)
    #         if cycle == total_cycles:
    #             indexes_to_query = unlabelled_index_list[:budget_last_cycle]
    #             unlabelled_index_list = unlabelled_index_list[budget_per_cycle:]
    #             budget_used = budget_used + budget_last_cycle
    #             budget = budget - budget_last_cycle
    #         else:
    #             indexes_to_query = unlabelled_index_list[:budget_per_cycle]
    #             unlabelled_index_list = unlabelled_index_list[budget_per_cycle:]
    #             budget_used = budget_used + budget_per_cycle
    #             budget = budget - budget_per_cycle
    #         labelled_index_list.extend(indexes_to_query)

    #         print("--------------- querying victim model ---------------")
    #         thief_dataset_train_for_victim = thief_dataset.get_loaded_features(index = index_to_query , split = "train" , features = thief_train_features_for_victim_model)
    #         thief_train_dataloader_for_victim = DataLoader(thief_dataset_train_for_victim, sampler=SequentialSampler(thief_dataset_train_for_victim), batch_size=BATCH_SIZE)
    #         true_train_labels_thief_sub , entropy_train_thief_sub , index_list_train_sub = query_data_in_victim(cfg=cfg, theif_dataloader=thief_train_dataloader_for_victim, victim_model=victim_model, victim_tokenizer=victim_tokenizer, victim_config=victim_config)
    #         true_train_labels_thief.update(true_train_labels_thief_sub)
    #         entropy_train_thief.update(entropy_train_thief_sub)
    #         index_list_train.extend(list(index_list_train_sub))

    
    # # with open(os.path.join(model_out_dir, "validation_results.txt"), "w") as f:
    # #     f.write("cycle\tglobal_step\ttraining_loss\tvalidation_accuracy\n")
    # #     for key in validation_accuracy_dict_all_models.keys():
    # #         for i, dicts in enumerate(validation_accuracy_dict_all_models[key]):
    # #             key2 = list(dicts.keys())[0]
    # #             f.write("%s\t%s\t%s\t%s\t" % (key2, global_step_dict_all_models[key][i][key2], loss_dict_all_models[key][i][key2], validation_accuracy_dict_all_models[key][i][key2]))

    # print("--------------- thief model setup for testing ---------------")
    # victim_dataset_test_for_victim = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset.get_features(split = 'test', tokenizer=victim_tokenizer, label_list=victim_dataset.get_labels()))
    # victim_test_dataloader_for_victim = DataLoader(victim_dataset_test_for_victim, sampler=SequentialSampler(victim_dataset_test_for_victim), batch_size=BATCH_SIZE)
    # result_victim = evaluate(cfg, victim_test_dataloader_for_victim, victim_model, victim_tokenizer, victim_config)
    # print("victim accuracy", result_victim)

    # thief_test_loader_per_model = {}
    # for key , tokenizer in list_tokenizers.items():
    #     victim_dataset_test_for_thief = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset.get_features(split = 'test', tokenizer=tokenizer, label_list=victim_dataset.get_labels()))
    #     victim_test_dataloader_for_thief = DataLoader(victim_dataset_test_for_thief, sampler=SequentialSampler(victim_dataset_test_for_thief), batch_size=BATCH_SIZE)
    #     thief_test_loader_per_model[key] = victim_test_dataloader_for_thief

    # metrics_list = {}

    # for key in list_models.keys():
    #     metrics_list[key] = []
    
    # for key in list_models.keys():
    #     print("Evaluating thief_model:", key)
    #     sub_model_out_dir = os.path.join(model_out_dir, key)
    #     list_models = os.listdir(sub_model_out_dir)
    #     list_models = [x for x in list_models if ".txt" not in x]
    #     list_models.sort()
    #     list_model_paths = [os.path.join(sub_model_out_dir, x) for x in list_models]
    #     print(list_models)
    #     for i , model_folder in enumerate(list_model_paths):
    #         eval_model = AutoModelForSequenceClassification.from_pretrained(model_folder, num_labels=cfg.VICTIM.NUM_LABELS)
    #         eval_tokenizer = AutoTokenizer.from_pretrained(model_folder)
    #         eval_config = AutoConfig.from_pretrained(model_folder)

    #         result_thief = evaluate(cfg, thief_test_loader_per_model[key], eval_model, eval_tokenizer, eval_config)
    #         metrics_list[key].append({list_models[i] : metrics(result_thief['true_labels'],  result_thief['preds'] , result_victim['preds'])})
    
    # print("metrics - ", metrics_list)

    # with open(os.path.join(model_out_dir, "metrics_logs.json"), "w") as f:
    #     json.dump(metrics_list, f)

    # # with open(os.path.join(model_out_dir, "accuracy_list.txt"), "w") as f:
    # #     f.write("model\tagreement_score\taccuracy\n")
    # #     for key in accuracy_list.keys():
    # #         for i, dicts in enumerate(accuracy_list[key]):
    # #             key2 = dicts.keys()
    # #             key2 = list(key2)
    # #             key2 = key2[0]
    # #             f.write("{}\t{}\t{}".format(key2, accuracy_list[key][i][key2], accuracy_list[key][i][key2]))
    # #             f.write("\n")
