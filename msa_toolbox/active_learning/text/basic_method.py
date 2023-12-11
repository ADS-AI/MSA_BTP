import os
import json
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

def all_data_stealing(cfg , victim_dataset, thief_dataset, victim_model, thief_model, victim_tokenizer, thief_tokenizer, victim_config, thief_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    BATCH_SIZE = cfg.THIEF_HYPERPARAMETERS.PER_GPU_BATCH_SIZE * max(1, n_gpu)
    set_seed(cfg, n_gpu)
    
    model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR, "thief-{}_victim-{}_thiefModel-{}_victimModel-{}_method-{}_epochs-{}_budget-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.ARCHITECTURE , cfg.VICTIM.ARCHITECTURE , cfg.ACTIVE.METHOD , cfg.THIEF_HYPERPARAMETERS.EPOCH, cfg.ACTIVE.BUDGET))
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    thief_model.to(device)  
    
    print("--------------- thief model setup for training ---------------")
    thief_dataset_train_features_for_thief = thief_dataset.get_features(split = 'train', tokenizer=thief_tokenizer, label_list=thief_dataset.get_labels())
    thief_dataset_train_all = thief_dataset.get_loaded_features(index = None , split = "train" , features = thief_dataset_train_features_for_thief)
    thief_train_dataloader_all = DataLoader(thief_dataset_train_all, sampler=RandomSampler(thief_dataset_train_all, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)

    # Dataset to query thief data in victim model and querying data in victim model 
    thief_dataset_train_for_victim = thief_dataset.get_loaded_features(index = None , split = "train" , features = thief_dataset.get_features(split = 'train', tokenizer=victim_tokenizer, label_list=thief_dataset.get_labels()))
    thief_train_dataloader_for_victim = DataLoader(thief_dataset_train_for_victim, sampler=RandomSampler(thief_dataset_train_for_victim, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    true_train_labels_thief , entropy_train_thief , index_list_train = query_data_in_victim(cfg=cfg, theif_dataloader=thief_train_dataloader_for_victim, victim_model=victim_model, victim_tokenizer=victim_tokenizer, victim_config=victim_config)

    # make train and validation split
    np.random.shuffle(index_list_train)
    train_indexes = index_list_train[:int(0.8 * len(index_list_train))]
    validation_indexes = index_list_train[int(0.8 * len(index_list_train)):]

    # Dataset for training and validation of thief model
    thief_dataset_train = thief_dataset.get_loaded_features(index = train_indexes , split = "train", true_labels = true_train_labels_thief , features = thief_dataset_train_features_for_thief)
    thief_dataset_validation = thief_dataset.get_loaded_features(index = validation_indexes , split = "train", true_labels = true_train_labels_thief, features = thief_dataset_train_features_for_thief)
    thief_train_dataloader = DataLoader(thief_dataset_train, sampler=RandomSampler(thief_dataset_train, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)
    thief_validation_dataloader = DataLoader(thief_dataset_validation, sampler=RandomSampler(thief_dataset_validation, generator=torch.Generator().manual_seed(cfg.SEED)), batch_size=BATCH_SIZE)

    # Train thief model
    global_step, tr_loss , validation_accuracy , thief_model , thief_config , thief_tokenizer = active_train(cfg , thief_train_dataloader , thief_validation_dataloader , thief_model , thief_config , thief_tokenizer)
    save_thief_model(thief_model , thief_tokenizer , thief_config, model_out_dir + "/{}".format(cfg.THIEF.ARCHITECTURE))

    # with open(os.path.join(model_out_dir, "validation_results.txt"), "w") as f:
    #     f.write("cycle_num\tglobal_step\ttraining_loss\tvalidation_accuracy\n")
    #     f.write("%s\t%s\t%s\t%s\t" % (1, global_step, tr_loss, validation_accuracy))

    print("global_step = %s, average_training_loss = %s , validation_accuracy = %s" % (global_step, tr_loss , validation_accuracy))

    print("--------------- thief model setup for testing ---------------")
    victim_dataset_test_for_victim = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset.get_features(split = 'test', tokenizer=victim_tokenizer, label_list=victim_dataset.get_labels()))
    victim_dataset_test_for_thief = victim_dataset.get_loaded_features(index = None , split = "test" , features = victim_dataset.get_features(split = 'test', tokenizer=thief_tokenizer, label_list=victim_dataset.get_labels()))
    victim_test_dataloader_for_victim = DataLoader(victim_dataset_test_for_victim, sampler=SequentialSampler(victim_dataset_test_for_victim), batch_size=BATCH_SIZE)
    victim_test_dataloader_for_thief = DataLoader(victim_dataset_test_for_thief, sampler=SequentialSampler(victim_dataset_test_for_thief), batch_size=BATCH_SIZE)

    # Evaluate victim model on victim test data
    print("Evaluating victim model on victim test data")
    result_victim  = evaluate(cfg, victim_test_dataloader_for_victim, victim_model, victim_tokenizer, victim_config)
    print("result_victim - ", result_victim)

    metrics_list = {}
    model_folders = os.listdir(model_out_dir)
    model_folders = [x for x in model_folders if ".txt" not in x]
    model_folders = [x for x in model_folders if ".json" not in x]
    model_folders.sort()
    model_folders_paths =  [os.path.join(model_out_dir, x) for x in model_folders]
    print(model_folders)
    for i , model_folder in enumerate(model_folders_paths):
        print("Evaluating thief_model:", model_folder)
        eval_model = AutoModelForSequenceClassification.from_pretrained(model_folder, num_labels=cfg.VICTIM.NUM_LABELS)
        eval_tokenizer = AutoTokenizer.from_pretrained(model_folder)
        eval_config = AutoConfig.from_pretrained(model_folder)

        # Evaluate thief model on victim test data
        result_thief= evaluate(cfg, victim_test_dataloader_for_thief, eval_model, eval_tokenizer, eval_config)
        metrics_list[model_folders[i]] = metrics(result_thief['true_labels'],  result_thief['preds'] , result_victim['preds'])

    print("metrics - ", metrics_list)

    # Save metrics
    with open(os.path.join(model_out_dir, "metrics_logs.json"), "w") as f:
        json.dump(metrics_list, f)
        
 