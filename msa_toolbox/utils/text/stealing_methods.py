import os
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from model_loader import load_untrained_model
from . cfg_reader import CfgNode

def active_learning_technique(cfg: CfgNode, theif_model: nn.Module, unlabeled_loader: DataLoader):
    if cfg.ACTIVE.METHOD == "qbc_stealing":
        return qbc_stealing(cfg, device, theif_model)
    elif cfg.ACTIVE.METHOD == "entroby_stealing":
        return entropy_stealing(cfg, theif_model, unlabeled_loader)
    elif cfg.ACTIVE.METHOD == "all_data_stealing":
        return all_data_stealing(cfg, theif_model, unlabeled_loader)

def Active_Learning_basic(cfg, n_gpu, device , prefix = ''):
    model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR,  "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(prefix) , cfg.THIEF.MODEL_TYPE)
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    validation_accuracy_dict = {}
    global_step_dict = {}
    loss_dict = {}

    config = AutoConfig.from_pretrained( cfg.THIEF.MODEL_NAME_OR_PATH,num_labels= cfg.VICTIM.NUM_LABELS ,finetuning_task=cfg.THIEF.TASK_NAME)
    tokenizer = AutoTokenizer.from_pretrained(cfg.THIEF.MODEL_NAME_OR_PATH,do_lower_case=cfg.THIEF.DO_LOWER_CASE,cache_dir=cfg.THIEF.CACHE_DIR if cfg.THIEF.CACHE_DIR else None)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.THIEF.MODEL_NAME_OR_PATH, from_tf=bool(".ckpt" in cfg.THIEF.MODEL_NAME_OR_PATH),config=config)
    model.to(device)  
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    model.save_pretrained(model_out_dir)
    tokenizer.save_pretrained(model_out_dir)
    config.save_pretrained(model_out_dir)
    print("--------------- model setup done ---------------")

    theif_datasets = Dataset.__dict__.keys()
    if cfg.THIEF.DATASET not in theif_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(theif_datasets))
    Thief_Dataset = Dataset.__dict__[cfg.THIEF.DATASET]
    Thief_Dataset = Thief_Dataset(cfg.THIEF.DATASET , cfg.THIEF.DATA_DIR, cfg.THIEF.NUM_LABELS , label_probs = False, Mode = cfg.THIEF.DATA_MODE , args = cfg)
    train_examples = Thief_Dataset.get_train_examples()
    train_label_list = Thief_Dataset.get_labels()
    train_features = convert_examples_to_features(train_examples,tokenizer,label_list=train_label_list,
        max_length=cfg.TRAIN.MAX_SEQ_LENGTH,
        pad_on_left=bool(cfg.VICTIM.MODEL_TYPE in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if cfg.VICTIM.MODEL_TYPE in ["xlnet"] else 0)
    train_data = load_and_cache_examples(cfg, train_features)
    sampler = RandomSampler(train_data)
    BATCH_SIZE = cfg.THIEF.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
    train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=BATCH_SIZE)
    
    if cfg.THIEF.DO_EVALUATION == True:
        val_examples = Thief_Dataset.get_dev_examples()
        val_label_list = Thief_Dataset.get_labels()
        val_features = convert_examples_to_features(val_examples,tokenizer,label_list=val_label_list,
                max_length=cfg.TRAIN.MAX_SEQ_LENGTH,
                pad_on_left=bool(cfg.THIEF.MODEL_TYPE in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if cfg.THIEF.MODEL_TYPE in ["xlnet"] else 0)
        val_data = load_and_cache_examples(cfg, val_features)
        sampler = RandomSampler(val_data)
        BATCH_SIZE = cfg.THIEF.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
        val_dataloader = DataLoader(val_data, sampler=sampler, batch_size=BATCH_SIZE)
    


    if cfg.THIEF.DO_TRAINING == True:
        print("---------------  quering victim model using theif data & created indexes ---------------")
        Thief_dataset_train , index_to_new_label_train, index_to_entropy_train , index_list_train =  query_all_data_theif(cfg=cfg, n_gpu=n_gpu, device=device , split='train')
        if cfg.THIEF.DO_EVALUATION == True:
            Thief_dataset_val , index_to_new_label_val, index_to_entropy_val , index_list_val =  query_all_data_theif(cfg=cfg, n_gpu=n_gpu, device=device , split='val')

        budget = cfg.ACTIVE.BUDGET
        unlabeled_index_list = list(index_list_train)
        labelled_index_list = []
        validation_index_list = []
        if True:
            unlabeled_index_list = random.sample(unlabeled_index_list, len(unlabeled_index_list))
            labelled_index_list.extend(unlabeled_index_list[0:1000])
            validation_index_list.extend(unlabeled_index_list[1000:2000])
            unlabelled_index_set = set(unlabeled_index_list) - set(labelled_index_list) - set(validation_index_list)
            unlabelled_index_list = list(unlabelled_index_set)

        budget = budget - 2000
        cycle_num = 1
        while(budget >= 0):
            print("--------------- cycle {} budget spent -{}---------------".format(cycle_num , budget))
            training_dataset = load_and_cache_examples(cfg , train_features, labelled_index_list)
            validation_dataset = load_and_cache_examples(cfg ,train_features ,validation_index_list)
            BATCH_SIZE_TRAIN = cfg.VICTIM.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
            BATCH_SIZE_EVAL = cfg.VICTIM.PER_GPU_EVAL_BATCH_SIZE * max(1, n_gpu)
            train_sampler = RandomSampler(training_dataset)
            train_loader = DataLoader(training_dataset, sampler=train_sampler, batch_size=BATCH_SIZE_TRAIN)
            validation_sampler = SequentialSampler(validation_dataset)
            validation_loader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=BATCH_SIZE_EVAL)
            global_step, tr_loss , validation_accuracy , model , config , tokenizer = active_train(cfg , train_loader , validation_loader , device , n_gpu , index_to_new_label_train, cycle_num , prefix)
            validation_accuracy_dict[cycle_num] = validation_accuracy
            global_step_dict[cycle_num] = global_step
            loss_dict[cycle_num] = tr_loss
            print("global_step = %s, average_training_loss = %s" % (global_step, tr_loss))
            print("validation accuracy", validation_accuracy)
            model.save_pretrained(model_out_dir)
            tokenizer.save_pretrained(model_out_dir)
            config.save_pretrained(model_out_dir)
            
            # select the top 1000 data
            if budget >= 1000:
                unlabelled_dataset = load_and_cache_examples(cfg , unlabelled_index_list , train_features)
                unlabelled_sampler = SequentialSampler(unlabelled_dataset)
                unlabelled_loader = DataLoader(unlabelled_dataset, sampler=unlabelled_sampler, batch_size=BATCH_SIZE_EVAL)
                print("--------------- get the entropy of the unlabelled data ---------------")
                index_to_entropy = get_entropy(cfg , model , tokenizer , config , n_gpu , device , unlabelled_loader)
                unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=True)
                labelled_index_list.extend(unlabelled_index_list[:1000])
                unlabelled_index_list = unlabelled_index_list[1000:]
                budget = budget - 1000
            elif budget == 0:
                break
            elif budget >= len(unlabelled_index_list) & budget < 1000:
                unlabelled_dataset = load_and_cache_examples(cfg , unlabelled_index_list , train_features)
                unlabelled_sampler = SequentialSampler(unlabelled_dataset)
                unlabelled_loader = DataLoader(unlabelled_dataset, sampler=unlabelled_sampler, batch_size=BATCH_SIZE_EVAL)
                print("--------------- get the entropy of the unlabelled data ---------------")
                index_to_entropy = get_entropy(cfg , model , tokenizer , config , n_gpu , device , unlabelled_loader)
                unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=True)
                budget = len(unlabelled_index_list)
                labelled_index_list.extend(unlabelled_index_list[:budget])  
                unlabelled_index_list = unlabelled_index_list[budget:]
                budget = -1
            else:
                unlabelled_dataset = load_and_cache_examples(cfg , unlabelled_index_list , train_features)
                unlabelled_sampler = SequentialSampler(unlabelled_dataset)
                unlabelled_loader = DataLoader(unlabelled_dataset, sampler=unlabelled_sampler, batch_size=BATCH_SIZE_EVAL)
                print("--------------- get the entropy of the unlabelled data ---------------")
                index_to_entropy = get_entropy(cfg , model , tokenizer , config , n_gpu , device , unlabelled_loader)
                unlabelled_index_list = sorted(unlabelled_index_list, key=lambda x: index_to_entropy[x], reverse=True)
                labelled_index_list.extend(unlabelled_index_list[:budget])
                unlabelled_index_list = unlabelled_index_list[budget:]
                budget = -1
            print("------------------- budget left -",budget, "-------------------")
            cycle_num = cycle_num + 1
    

    with open(os.path.join(model_out_dir, "validation_results.txt"), "w") as f:
        f.write("cycle_num\tglobal_step\ttraining_loss\tvalidation_accuracy\t")
        for cycle_num in validation_accuracy_dict.keys():
            f.write("%s\t%s\t%s\t%s\t" % (cycle_num, global_step_dict[cycle_num], loss_dict[cycle_num], validation_accuracy_dict[cycle_num]))
    print(validation_accuracy_dict)

    if cfg.THIEF.DO_TESTING == True:
        Dataset_list = Dataset.__dict__.keys()
        if cfg.VICTIM.DATASET not in Dataset_list:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(Dataset_list))
        Dataset_victim = Dataset.__dict__[cfg.VICTIM.DATASET]
        Dataset_victim = Dataset_victim(cfg.VICTIM.DATASET , cfg.VICTIM.DATA_DIR, cfg.VICTIM.NUM_LABELS , label_probs = False, Mode = cfg.VICTIM.DATA_MODE , args = cfg)
        if cfg.VICTIM.DATASET != 'sst2':
            victim_examples = Dataset_victim.get_test_examples()
        else:
            victim_examples = Dataset_victim.get_dev_examples()
        victim_label_list = Dataset_victim.get_labels()
        victim_features = convert_examples_to_features(victim_examples,tokenizer,label_list=victim_label_list,
            max_length=cfg.TRAIN.MAX_SEQ_LENGTH,
            pad_on_left=bool(cfg.THIEF.MODEL_TYPE in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if cfg.THIEF.MODEL_TYPE in ["xlnet"] else 0)
        victim_data = load_and_cache_examples(cfg, victim_features)
        sampler = RandomSampler(victim_data)
        BATCH_SIZE = cfg.THIEF.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
        victim_dataloader = DataLoader(victim_data, sampler=sampler, batch_size=BATCH_SIZE)

        accuracy_list = {}
        model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR,  "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(prefix) )
        model_folders = os.listdir(model_out_dir)
        model_folders.sort()
        model_folders = [os.path.join(model_out_dir, x) for x in model_folders]
        print(model_folders)
        for model_folder in model_folders:
            print("Evaluating model:", model_folder)
            model = AutoModelForSequenceClassification.from_pretrained(model_folder)
            tokenizer = AutoTokenizer.from_pretrained(model_folder)
            config = AutoConfig.from_pretrained(model_folder)
            model.to(device)
            result = evaluate(cfg, victim_dataloader, model, tokenizer, config, n_gpu , device)
            accuracy_list[model_folder] = result 
        print("accuracy list", accuracy_list)
        with open(os.path.join(model_out_dir, "accuracy_list.txt"), "w") as f:
            for key, value in accuracy_list.items():
                f.write("%s,%s" % (key, value))
                f.write("\n")
        return accuracy_list

def All_Data_Stealing(cfg , n_gpu , device , prefix = ""):
    config = AutoConfig.from_pretrained( cfg.THIEF.MODEL_NAME_OR_PATH,num_labels= cfg.VICTIM.NUM_LABELS ,finetuning_task=cfg.THIEF.TASK_NAME)
    tokenizer = AutoTokenizer.from_pretrained(cfg.THIEF.MODEL_NAME_OR_PATH,do_lower_case=cfg.THIEF.DO_LOWER_CASE,cache_dir=cfg.THIEF.CACHE_DIR if cfg.THIEF.CACHE_DIR else None)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.THIEF.MODEL_NAME_OR_PATH, from_tf=bool(".ckpt" in cfg.THIEF.MODEL_NAME_OR_PATH),config=config)
    model.to(device)

    theif_datasets = Dataset.__dict__.keys()
    if cfg.THIEF.DATASET not in theif_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(theif_datasets))
    Thief_Dataset = Dataset.__dict__[cfg.THIEF.DATASET]
    Thief_Dataset = Thief_Dataset(cfg.THIEF.DATASET , cfg.THIEF.DATA_DIR, cfg.THIEF.NUM_LABELS , label_probs = False, Mode = cfg.THIEF.DATA_MODE , args = cfg)
    train_examples = Thief_Dataset.get_train_examples()
    train_label_list = Thief_Dataset.get_labels()
    train_features = convert_examples_to_features(train_examples,tokenizer,label_list=train_label_list,
        max_length=cfg.TRAIN.MAX_SEQ_LENGTH,
        pad_on_left=bool(cfg.VICTIM.MODEL_TYPE in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if cfg.VICTIM.MODEL_TYPE in ["xlnet"] else 0)
    train_data = load_and_cache_examples(cfg, train_features)
    sampler = RandomSampler(train_data)
    BATCH_SIZE = cfg.THIEF.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
    train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=BATCH_SIZE)
    
    if cfg.THIEF.DO_EVALUATION == True:
        val_examples = Thief_Dataset.get_dev_examples()
        val_label_list = Thief_Dataset.get_labels()
        val_features = convert_examples_to_features(val_examples,tokenizer,label_list=val_label_list,
                max_length=cfg.TRAIN.MAX_SEQ_LENGTH,
                pad_on_left=bool(cfg.THIEF.MODEL_TYPE in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if cfg.THIEF.MODEL_TYPE in ["xlnet"] else 0)
        val_data = load_and_cache_examples(cfg, val_features)
        sampler = RandomSampler(val_data)
        BATCH_SIZE = cfg.THIEF.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
        val_dataloader = DataLoader(val_data, sampler=sampler, batch_size=BATCH_SIZE)
    

    if cfg.THIEF.DO_TRAINING == True:
        print("---------------  quering victim model using theif data & created indexes ---------------")
        model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR, "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(prefix)  , cfg.THIEF.MODEL_TYPE)
        if not os.path.exists(model_out_dir):
            os.makedirs(model_out_dir)
        index_to_new_label_train, index_to_entropy_train , index_list_train =  query_all_data_theif(cfg=cfg, theif_dataloader=train_dataloader, n_gpu=n_gpu, device=device)
        if cfg.THIEF.DO_EVALUATION == True:
            index_to_new_label_val, index_to_entropy_val , index_list_val =  query_all_data_theif(cfg=cfg, theif_dataloader=val_dataloader, n_gpu=n_gpu, device=device)
        print("--------------- Train theif with all data ---------------")
        # prefix = ""
        if cfg.THIEF.DO_EVALUATION == True:
            global_step, tr_loss , model , tokenizer , config = train_theif(cfg,  train_dataloader, val_dataloader,  model, tokenizer, config , n_gpu , device , index_to_new_label=index_to_new_label_train, index_to_new_label_val = index_to_new_label_val , prefix = prefix)
        else:
            print("--------------- model setup ---------------") 
            global_step, tr_loss , model , tokenizer , config = train_theif(cfg,  train_dataloader, None,  model, tokenizer, config , n_gpu , device , index_to_new_label=index_to_new_label_train, prefix = prefix)
        
        print("Saving model to", model_out_dir)
        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(model_out_dir)
        tokenizer.save_pretrained(model_out_dir)
        torch.save(cfg, os.path.join(model_out_dir, "training_args.bin"))
        print("global_step = %s, average loss = %s" % (global_step, tr_loss))
   
    if cfg.THIEF.DO_TESTING == True:
        Dataset_list = Dataset.__dict__.keys()
        if cfg.VICTIM.DATASET not in Dataset_list:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(Dataset_list))
        Dataset_victim = Dataset.__dict__[cfg.VICTIM.DATASET]
        Dataset_victim = Dataset_victim(cfg.VICTIM.DATASET , cfg.VICTIM.DATA_DIR, cfg.VICTIM.NUM_LABELS , label_probs = False, Mode = cfg.VICTIM.DATA_MODE , args = cfg)
        if cfg.VICTIM.DATASET != 'sst2':
            victim_examples = Dataset_victim.get_test_examples()
        else:
            victim_examples = Dataset_victim.get_dev_examples()
        victim_label_list = Dataset_victim.get_labels()
        victim_features = convert_examples_to_features(victim_examples,tokenizer,label_list=victim_label_list,
            max_length=cfg.TRAIN.MAX_SEQ_LENGTH,
            pad_on_left=bool(cfg.THIEF.MODEL_TYPE in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if cfg.THIEF.MODEL_TYPE in ["xlnet"] else 0)
        victim_data = load_and_cache_examples(cfg, victim_features)
        sampler = RandomSampler(victim_data)
        BATCH_SIZE = cfg.THIEF.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
        victim_dataloader = DataLoader(victim_data, sampler=sampler, batch_size=BATCH_SIZE)

        accuracy_list = {}
        model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR,  "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(prefix))
        model_folders = os.listdir(model_out_dir)
        model_folders.sort()
        model_folders = [os.path.join(model_out_dir, x) for x in model_folders]
        print(model_folders)
        for model_folder in model_folders:
            print("Evaluating model:", model_folder)
            model = AutoModelForSequenceClassification.from_pretrained(model_folder)
            tokenizer = AutoTokenizer.from_pretrained(model_folder)
            config = AutoConfig.from_pretrained(model_folder)
            model.to(device)
            result = evaluate(cfg, victim_dataloader, model, tokenizer, config, n_gpu , device)
            accuracy_list[model_folder] = result 
        print("accuracy list", accuracy_list)
        with open(os.path.join(model_out_dir, "accuracy_list.txt"), "w") as f:
            for key, value in accuracy_list.items():
                f.write("%s,%s" % (key, value))
                f.write("\n")
  
def qbc_stealing(cfg : CfgNode , device: torch.device() , theif_model: nn.Module , list_thief_models : list):
    validation_accuracy_dict_all_models = [{}]*cfg.ACTIVE.NUM_MODELS
    global_step_dict_all_models = [{}]*cfg.ACTIVE.NUM_MODELS
    loss_dict_all_models = [{}]*cfg.ACTIVE.NUM_MODELS

    thief_models = {}
    thief_config = {}
    thief_tokenizer = {}

    if cfg.THIEF.DO_TRAINING == True:
        for i in range(cfg.ACTIVE.NUM_MODELS):
            theif_model[i] , thief_config[i] , thief_tokenizer[i] = load_untrained_model(cfg, list_thief_models[i])


    if cfg.THIEF.DO_TRAINING == True:
        print("---------------  quering victim model using theif data & created indexes ---------------")
        Thief_dataset_train , index_to_new_label_train, index_to_entropy_train , index_list_train =  query_all_data_theif(cfg=cfg, n_gpu=n_gpu, device=device , split='train')
        if cfg.THIEF.DO_EVALUATION == True:
            Thief_dataset_val , index_to_new_label_val, index_to_entropy_val , index_list_val =  query_all_data_theif(cfg=cfg, n_gpu=n_gpu, device=device , split='val') 
        budget = cfg.ACTIVE.BUDGET
        unlabeled_index_list = list(index_list_train)
        labelled_index_list = []
        validation_index_list = []
        if True:
            unlabeled_index_list = random.sample(unlabeled_index_list, len(unlabeled_index_list))
            labelled_index_list.extend(unlabeled_index_list[0:1000])
            validation_index_list.extend(unlabeled_index_list[1000:2000])
            unlabelled_index_set = set(unlabeled_index_list) - set(labelled_index_list) - set(validation_index_list)
            unlabelled_index_list = list(unlabelled_index_set)
        budget = budget - 2000
        cycle_num = 1
        examples = Thief_dataset_train.get_train_examples()
        label_list = Thief_dataset_train.get_labels()
        print("-----Creating features from dataset-----")
        features = convert_examples_to_features(examples,thief_tokenizer,label_list=label_list,
            max_length=cfg.TRAIN.MAX_SEQ_LENGTH,
            pad_on_left=bool(cfg.VICTIM.MODEL_TYPE in ["xlnet"]),  # pad on the left for xlnet
            pad_token=thief_tokenizer.convert_tokens_to_ids([thief_tokenizer.pad_token])[0],
            pad_token_segment_id=4 if cfg.VICTIM.MODEL_TYPE in ["xlnet"] else 0)
        print("-----features created")
        index_to_entropy = {}
        index_to_label_prob = {}

        while(budget >= 0):
            for i in range(num_models):
                model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR,  "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(i)  , cfg.THIEF.MODEL_TYPE)
                print("--------------- cycle {} budget spent -{}---------------".format(cycle_num , budget))
                training_dataset = load_and_cache_examples_by_index(cfg , labelled_index_list , features)
                validation_dataset = load_and_cache_examples_by_index(cfg , validation_index_list , features)
                BATCH_SIZE_TRAIN = cfg.VICTIM.PER_GPU_TRAIN_BATCH_SIZE * max(1, n_gpu)
                BATCH_SIZE_EVAL = cfg.VICTIM.PER_GPU_EVAL_BATCH_SIZE * max(1, n_gpu)
                train_sampler = SequentialSampler(training_dataset)
                train_dataloader = DataLoader(training_dataset, sampler=train_sampler, batch_size=BATCH_SIZE_TRAIN)
                validation_sampler = SequentialSampler(validation_dataset)
                validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=BATCH_SIZE_EVAL)
                global_step, tr_loss , validation_accuracy , model , config , tokenizer = active_train(cfg , train_dataloader , validation_dataloader , device , n_gpu , index_to_new_label_train, cycle_num , i)
                validation_accuracy_dict_all_models[i][cycle_num] = validation_accuracy
                global_step_dict_all_models[i][cycle_num] = global_step
                loss_dict_all_models[i][cycle_num] = tr_loss
                print("global_step = %s, average_training_loss = %s" % (global_step, tr_loss))
                print("validation accuracy", validation_accuracy)
                model.save_pretrained(model_out_dir)
                tokenizer.save_pretrained(model_out_dir)
                config.save_pretrained(model_out_dir)
            
            if budget > 0:
                unlabelled_dataset = load_and_cache_examples_by_index(cfg , unlabelled_index_list , features)
                unlabelled_sampler = SequentialSampler(unlabelled_dataset)
                unlabelled_dataloader = DataLoader(unlabelled_dataset, sampler=unlabelled_sampler, batch_size=BATCH_SIZE_EVAL)
                
                for i in range(num_models):
                    print("--------------- get the entropy of the unlabelled data ---------------")
                    model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR,  "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(i)  , cfg.THIEF.MODEL_TYPE)
                    model = AutoModelForSequenceClassification.from_pretrained(model_out_dir, num_labels=cfg.VICTIM.NUM_LABELS)
                    tokenizer = AutoTokenizer.from_pretrained(model_out_dir)
                    config = AutoConfig.from_pretrained(model_out_dir)
                    model.to(device)
                    # index_to_entropy[i] = get_entropy(cfg , model , tokenizer , config , n_gpu , device , unlabelled_dataloader)
                    index_to_label_prob[i] = get_label_probs(cfg , model , tokenizer , config , n_gpu , device , unlabelled_dataloader)
            
                unlabelled_index_list = get_index_by_vote(index_to_label_prob)

            if budget >= 1000:
                labelled_index_list.extend(unlabelled_index_list[:1000])
                unlabelled_index_list = unlabelled_index_list[1000:]
                budget = budget - 1000
            elif budget == 0:
                break
            elif budget >= len(unlabelled_index_list) & budget < 1000:
                budget = len(unlabelled_index_list)
                labelled_index_list.extend(unlabelled_index_list[:budget])  
                unlabelled_index_list = unlabelled_index_list[budget:]
                budget = -1
            else:
                labelled_index_list.extend(unlabelled_index_list[:budget])
                unlabelled_index_list = unlabelled_index_list[budget:]
                budget = -1
            print("------------------- budget left -",budget, "-------------------")
            cycle_num = cycle_num + 1


        with open(os.path.join(model_out_dir, "validation_results.txt"), "w") as f:
            for i in range(10):
                for key in validation_accuracy_dict_all_models[i]:
                    f.write("model {} cycle {} validation accuracy: {} ".format(i, key, validation_accuracy_dict_all_models[i][key]))

    if cfg.THIEF.DO_TESTING == True:
          # get all folders in model_out_dir
        for i in range(num_models):
                accuracy_list = {}
                model_out_dir = os.path.join(cfg.THIEF_MODEL_DIR,  "thief-{} victim-{} thief_model-{} method-{}".format(cfg.THIEF.DATASET, cfg.VICTIM.DATASET , cfg.THIEF.MODEL_TYPE , cfg.ACTIVE.METHOD), "prefix-{}".format(i) )
                # model = AutoModelForSequenceClassification.from_pretrained(model_out_dir, num_labels=cfg.VICTIM.NUM_LABELS)
                # tokenizer = AutoTokenizer.from_pretrained(model_out_dir)
                # config = AutoConfig.from_pretrained(model_out_dir)
                # model.to(device)
                model_folders = os.listdir(model_out_dir)
                # sort the model folders alphabetically
                model_folders.sort()
                model_folders = [os.path.join(model_out_dir, x) for x in model_folders]
                print(model_folders)
                for model_folder in model_folders:
                    print(model_folder)
                    model = AutoModelForSequenceClassification.from_pretrained(model_folder)
                    tokenizer = AutoTokenizer.from_pretrained(model_folder)
                    config = AutoConfig.from_pretrained(model_folder)
                    model.to(device)
                    victim_name = cfg.VICTIM.DATASET
                    Dataset_list = Dataset.__dict__.keys()
                    if victim_name not in Dataset_list:
                        raise ValueError('Dataset not found. Valid arguments = {}'.format(Dataset_list))
                    Dataset_victim = Dataset.__dict__[victim_name]
                    Dataset_victim = Dataset_victim(cfg.VICTIM.DATASET , cfg.VICTIM.DATA_DIR, cfg.VICTIM.NUM_LABELS , label_probs = False, Mode = cfg.VICTIM.DATA_MODE , args = cfg)
                    if cfg.VICTIM.DATA_MODE == "train":
                        victim_dataset = Dataset_victim.get_train_examples()
                    if cfg.VICTIM.DATASET == 'sst2':
                        result = evaluate_thief(cfg, Dataset_victim, model, tokenizer, config, n_gpu , device, split='dev')
                    else:
                        result = evaluate_thief(cfg, Dataset_victim, model, tokenizer, config, n_gpu , device, split='test')
                    accuracy_list[model_folder] = result
                print("accuracy list", accuracy_list)
                # save the accuracy list to txt file
                with open(os.path.join(model_out_dir, "accuracy_list.txt"), "w") as f:
                    for key, value in accuracy_list.items():
                        f.write("%s,%s" % (key, value))
                        f.write("\n")