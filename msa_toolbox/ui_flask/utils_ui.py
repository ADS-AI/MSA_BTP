import yaml
import os

def extract_data(form):
    print(form)

    name = form['config_name']
    batch_size = form['batch_size']
    
    v_dataset = form['V_data']
    v_model = form['V_arch']
    # v_api = form['V_api']
    t_dataset = form['T_data']
    t_model = form['T_arch']
    # subset = form['subset']
    
    Method = form['method']
    Budget = float(form['budget'])
    Optimizer = form['optim']
    Criteria = form['criteria']
    
    Device = form['device']
    Epochs = float(form['Epochs'])
    # to be added
    Cycles = float(form['Cycles'])
    # Patience = form['Patience']
    # log_dir = form['log_dir']
    out_dir = form['out_dir']
    v_data_root = form['V_data_root']
    t_data_root = form['t_data_root']
    feature = form['feature']
    metric = form['metric']
    DFAL_MAX_ITER = int(form['DFAL_MAX_ITER'])
    K = int(form['K'])
    NUM_VAE_STEPS = int(form['NUM_VAE_STEPS'])
    NUM_ADV_STEPS = int(form['NUM_ADV_STEPS'])
    ADVERSARY_PARAM = int(form['ADVERSARY_PARAM'])
    BETA = int(form['beta_img']) 
    
    victim={'DATASET':v_dataset,
            'ARCHITECTURE':v_model,
            'DATA_ROOT':v_data_root,
            'WEIGHTS':'default',
            'DEFENCE':form['defence'],
            'SHAPIRO_THRESHOLD':0.994,
            'NUM_CLASSES':int(form['out_label']),
            'MSP_THRESHOLD':0.5,
            'IS_API': bool(form['is_api']),
            }
    
    thief={'DATASET':t_dataset,
           'ARCHITECTURE':t_model,
           'DATA_ROOT':t_data_root,
           'SUBSET':20000,
           'NUM_TRAIN':100000,
           'WEIGHTS':'default'}
  
    active={'BUDGET':Budget,
            'METHOD':Method,
            'CYCLES':Cycles,
            'FEATURE':feature,
            'METRIC':metric,
            'DFAL_MAX_ITER':DFAL_MAX_ITER,
            'K':K,
            'NUM_VAE_STEPS':NUM_VAE_STEPS,
            'NUM_ADV_STEPS':NUM_ADV_STEPS,
            'ADVERSARY_PARAM':ADVERSARY_PARAM,
            'BETA':BETA,}    
    
    train={'OPTIMIZER':Optimizer,
           'LOSS_CRITERION':Criteria,
           'BATCH_SIZE':batch_size,
           'WEIGHT_DECAY':0.0001,
           'EPOCH':Epochs,
           'PATIENCE':5,
           'LR':float(form['LR']),
           'BLACKBOX_TRAINING':bool(form['bbt']),
           'LOG_INTERVAL':10,
           'MILESTONES':[30,80],
           }
    
    cfg ={'VICTIM':victim,
          'THIEF':thief,
          'ACTIVE':active,
          'TRAIN':train,
          'TRIALS':1,
          'DS_SEED':123,
          'NUM_WORKERS':2, 
          'DEVICE':Device+":0",
          'OUT_DIR':out_dir,
          }
    # print(cfg)

    # yaml_string=yaml.dump(cfg, default_flow_style=False,sort_keys=False)
    # print("The YAML string is:")
    # print(yaml_string)
    
    
    #save the yaml file to the disk 
    path = os.path.join(os.getcwd(), 'msa_toolbox/ui_flask/configs/image/'+name+'.yaml')
    yaml.dump(cfg, open(path, 'w'),sort_keys=False)
    return 'config file generated'


def extract_data_text(form_data):
    name = form_data['config_name_text']
    yaml_data = {
        "VICTIM": {
            "DATASET": form_data.get('V_data_text', ''),
            "ARCHITECTURE": form_data.get('V_arch_text', ''),
            "DATA_ROOT": "",
            "DATA_MODE": form_data.get('V_data_mode_text', ''),
            "NUM_LABELS": int(form_data.get('out_label', '0'))
        },
        "THIEF": {
            "DATASET": form_data.get('T_data_text', ''),
            "ARCHITECTURE": form_data.get('T_arch_text', ''),
            "DATA_ROOT": "",
            "DATA_MODE": form_data.get('T_data_mode_text', ''),
            "NUM_LABELS": int(form_data.get('T_out_label_text', '0'))
        },
        "ACTIVE": {
            "USE_PRETRAINED": form_data.get('Pretrained', '').lower() == 'on',
            "PRETRAINED_PATH": form_data.get('path_pretrained', '') if form_data.get('Pretrained', '').lower() == 'on' else None,
            "BUDGET": int(form_data.get('budget_text', '0')),
            "METHOD": form_data.get('method', ''),
            "AUGMENTATION": ['None'],
            "CUTMIX_PROB": float(form_data.get('cutmix_prob', '0.0')),
            "BETA": float(form_data.get('beta', '0.0')),
            "VAL": int(form_data.get('val', '0')),
            "INITIAL": int(form_data.get('Initial', '0')),
            "CYCLES": int(form_data.get('Cycles', '0')),
            "ADDENUM": int(form_data.get('ADDENUM', '0')),
            "ALPHA": float(form_data.get('Alpha', '0.0'))
        },
        "THIEF_HYPERPARAMETERS": {
            "BATCH": int(form_data.get('batch_size_text', '0')),
            "OPTIMIZER": form_data.get('optim', '').lower(),
            "LR": float(form_data.get('lr_text', '0.0')),
            "MOMEMTUM": float(form_data.get('momentum_text', '0.0')),
            "WDECAY": float(form_data.get('wdecay', '0.0')),
            "EPOCH": int(form_data.get('Epochs', '0')),
            "MILESTONES": list([int(step) for step in form_data.get('lr_epochl', '').split(',')] if form_data.get('lr_epochl', '') else []),
            "GAMMA": float(form_data.get('gamma_text', '0.0')),
            "ADAM_EPS": float(form_data.get('adam_eps', '0.0')),
            "WARMUP_STEPS": int(form_data.get('w_up_text', '0')),
            "LR_MARGIN": float(form_data.get('lr_margin', '0.0')),
            "LR_WEIGHT": float(form_data.get('lr_weight', '0.0')),
            "LR_EPOCHL": int(form_data.get('lr_epochl', '0'))
        },
        "TRIALS": int(form_data.get('trials-text', '0')),
        "LOCAL_RANK": int(form_data.get('Local-Rank-text', '0')),
        "RNG_SEED": int(form_data.get('rng-seed-text', '0')),
        "DS_SEED": int(form_data.get('Ds-seed-text', '0')),
        "NUM_WORKERS": int(form_data.get('no-workers-text', '0')),
        "CACHE_DIR": form_data.get('out_dir-text', '') + "/cache",
        "DEVICE": form_data.get('device', ''),
        "LOG_DEST": form_data.get('out_dir-text', '') + '/logs',
        "OUT_DIR": form_data.get('out_dir-text', ''),
        "VICTIM_MODEL_DIR": form_data.get('out_dir-text', '') + "/victim_models",
        "THIEF_MODEL_DIR": form_data.get('out_dir-text', '') + "/thief_models"
    }

#     with open('output.yaml', 'w') as yaml_file:
#         yaml.dump(yaml_data, yaml_file, default_flow_style=False)
        
    path = os.path.join(os.getcwd(), 'msa_toolbox/ui_flask/configs/text/'+name+'.yaml')
    yaml.dump(yaml_data, open(path, 'w'),sort_keys=False,default_flow_style=False)
    return 'config file generated'

