from flask import Flask, render_template, request , flash ,jsonify
# from flask_socketio import SocketIO, emit
import yaml
import os
import json
from utils_ui import extract_data as extract_data_image

app = Flask(__name__)
app.secret_key = 'some_secret_key'
current_dir = os.path.dirname(os.path.abspath(__file__))
path_log = os.path.join(current_dir, 'logs/log.txt')
path_json = os.path.join(current_dir, 'logs/log_metrics.json')


datasets =['cifar10','cifar100','imagenet','mnist','kmnist','fashionmnist','emnist','emnistletters','svhn','tinyimagenet200','tinyimagesubset','cubs200',
             'diabetic5','indoor67','caltech256']

archi = ['resnet18','resnet50','vgg11','vgg13','vgg16','vgg19','vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn','alexnet','efficientnet_b0','efficientnet_b1',
             'efficientnet_b2','efficientnet_b3','efficientnet_b4','efficientnet_b5','efficientnet_b6','efficientnet_b7','efficientnet_v2_s','efficientnet_v2_m','efficientnet_v2_l',
             'mobilenet_v2','mobilenet_v3_large','mobilenet_v3_small']

methods = ['random','entropy']

optimizers = ['adam','sgd','rmsprop','adagrad','adadelta','adamax']

criterias = ['cross_entropy_loss','mse_loss','l1_loss','soft_margin_loss','bce_loss']

options = {'datasets': datasets, 'archi': archi, 'methods': methods, 'optimizers': optimizers, 'criterias': criterias}
    
    
    
@app.route('/file_content')
def get_file_progress():
    with open(path_log) as f:
        content = f.read()
        return content
    


@app.route('/training', methods=['GET','POST'])
def tranning():
    if request.method == 'POST':
        print(request.form)
        return render_template('progress.html',configs=[],fg=fg)
    else:
        configs = [os.listdir('msa_toolbox/ui_flask/configs/image'),os.listdir('msa_toolbox/ui_flask/configs/text')]
    if fg==1:
        return render_template('progress.html',configs=configs,active = 'traning')
    return render_template('index.html', configs=configs,active = 'traning')

@app.route('/')
def index():
    return render_template('index1.html',options=options,active = 'home')


@app.route('/config_image', methods=['GET','POST'])
def submit():
    if request.method == 'POST':
        msg = extract_data(dict(request.form))
        
        flash(msg)
        return render_template('index1.html', options=options)
    return render_template('index1.html', options=options)


# # def extract_data(form):
#     print(form)

#     name = form['config_name']
#     batch_size = form['batch_size']
    
#     v_dataset = form['V_data']
#     v_model = form['V_arch']
    
#     t_dataset = form['T_data']
#     t_model = form['T_arch']
#     # subset = form['subset']
    
#     Method = form['method']
#     Budget = float(form['budget'])
#     Optimizer = form['optim']
#     Criteria = form['criteria']
    
#     Device = form['device']
#     Epochs = float(form['Epochs'])
#     # to be added
#     Cycles = float(form['Cycles'])
#     # Patience = form['Patience']
#     log_dir = form['log_dir']
#     out_dir = form['out_dir']
#     v_data_root = form['v_data_root']
#     t_data_root = form['t_data_root']
    
#     victim={'DATASET':v_dataset,
#             'ARCHITECTURE':v_model,
#             'DATA_ROOT':v_data_root,
#             'WEIGHTS':'default'}
    
#     thief={'DATASET':t_dataset,
#            'ARCHITECTURE':t_model,
#            'DATA_ROOT':t_data_root,
#            'SUBSET':20000,
#            'NUM_TRAIN':100000,
#            'WEIGHTS':'default'}
  
#     active={'BUDGET':Budget,
#             'METHOD':Method,
#             'CYCLES':Cycles}    
    
#     train={'OPTIMIZER':Optimizer,
#            'LOSS_CRITERION':Criteria,
#            'BATCH_SIZE':128,
#            'WEIGHT_DECAY':0.0001,
#            'EPOCH':Epochs,
#            'PATIENCE':5,
#            'LR':0.001,
#            }
    
#     cfg ={'VICTIM':victim,
#           'THIEF':thief,
#           'ACTIVE':active,
#           'TRAIN':train,
#           'TRIALS':1,
#           'DS_SEED':123,
#           'NUM_WORKERS':2, 
#           'DEVICE':Device,
#           'LOG_DEST':log_dir,
#           'OUT_DEST':out_dir,
#           }
#     # print(cfg)

#     # yaml_string=yaml.dump(cfg, default_flow_style=False,sort_keys=False)
#     # print("The YAML string is:")
#     # print(yaml_string)
    
    
#     #save the yaml file to the disk 
#     path = os.path.join(os.getcwd(), 'msa_toolbox/ui_flask/configs/image/'+name+'.yaml')
#     yaml.dump(cfg, open(path, 'w'),sort_keys=False)
#     return 'config file generated'

def extract_data(form):
    return extract_data_image(form)
    

@app.route('/chart')
def chart():
    # metric_path = os.path.join(os.getcwd(), 'msa_toolbox/ui_flask/logs/log_metrics.json')
    with open(path_json) as f:
        data = json.load(f)
    labels = data.keys()
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
    
    # return the chart data as a JSON object
    return jsonify({'accuracy_victim': accuracy_victim, 'accuracy_thief': accuracy_thief, 'precision_victim': precision_victim, 'precision_thief': precision_thief, 'f1_victim': f1_victim, 'f1_thief': f1_thief, 'agreement_victim': agreement_victim, 'agreement_thief': agreement_thief, 'labels': list(labels)})


# @app.route('/chart_draw')
# def heloijh():
#     return render_template('chart.html')


@app.route('/progress')
def progress():
    global fg
    if fg==1:
        return render_template('progress.html',fg=fg)
    else:
        return render_template('progress.html')

@app.route('/update_image', methods=['POST'])
def pr():
    filename = current_dir+"/configs/image/"+request.form['config_file_name']
    print(filename)
    file = yaml.load(open(filename), Loader=yaml.FullLoader)
    print(file)
    return render_template('update.html',options=options,active = 'traning',file=file)


@app.route('/testing', methods=['GET','POST'])
def test():
    return render_template('test_index.html',options=options,active = 'testing')
@app.route('/get_existing_config_files', methods=['GET'])
def get_existing_config_files():
    # Logic to retrieve a list of existing config file names
    # Return the list as JSON
    return jsonify(os.listdir('msa_toolbox/ui_flask/configs/image'))

@app.route('/get_config_data', methods=['GET'])
def get_config_data():
    selected_config_file = request.args.get('selected_config_file')
    if selected_config_file == 'create_new':
        # Return an empty structure for a new config file
        return jsonify({'config_data': {}})

    # Logic to fetch the data of the selected config file
    selected_config_file_data = yaml.load(open('msa_toolbox/ui_flask/configs/image/'+selected_config_file), Loader=yaml.FullLoader)
    # Return the data as JSON
    print(selected_config_file_data)
    return jsonify({'config_data': dict(selected_config_file_data)})

app.run(host='127.0.0.1', port=8085, debug=True)
