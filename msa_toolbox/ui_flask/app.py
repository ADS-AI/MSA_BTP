from flask import Flask, render_template, request , flash ,jsonify , redirect , url_for
# from flask_socketio import SocketIO, emit
import yaml
import os
import json
from utils_ui import extract_data as extract_data_image
from utils_ui import extract_data_text
from threading import Thread
# from msa_toolbox.main import app as image_app

app = Flask(__name__)
app.secret_key = 'some_secret_key'
current_dir = os.path.dirname(os.path.abspath(__file__))
path_log = os.path.join(current_dir, 'logs/log.txt')
path_json = os.path.join(current_dir, 'logs/log_metrics.json')

# fg=1

datasets =['cifar10','cifar100','imagenet','mnist','kmnist','fashionmnist','emnist','emnistletters','svhn','tinyimagenet200','tinyimagesubset','cubs200',
             'diabetic5','indoor67','caltech256']

archi = ['resnet18','resnet50','vgg11','vgg13','vgg16','vgg19','vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn','alexnet','efficientnet_b0','efficientnet_b1',
             'efficientnet_b2','efficientnet_b3','efficientnet_b4','efficientnet_b5','efficientnet_b6','efficientnet_b7','efficientnet_v2_s','efficientnet_v2_m','efficientnet_v2_l',
             'mobilenet_v2','mobilenet_v3_large','mobilenet_v3_small']

methods = ['random','entropy','vaal','kcenter','montecarlo','dfal']

optimizers = ['adam','sgd','rmsprop','adagrad','adadelta','adamax']

criterias = ['cross_entropy_loss','mse_loss','l1_loss','soft_margin_loss','bce_loss']

text_models = ["bert-base-uncased","roberta-base","xlnet-base-cased"]
datasets_text = ['yelp' , 'sst2' , 'pubmed' , 'twitter_finance' , 'twitter' , 'ag_news' , 'wiki_medical_terms', 'imdb' , 'mnli', 'boolq']
text_modes = ["HuggingFace","csv","tsv"]

defences  = ['None', 'prada','Adaptive-misinformation']
feature = ['avg_pool','fc']
metric = ['euclidean','cosine']
options = {'datasets': datasets, 'archi': archi, 'methods': methods, 'optimizers': optimizers, 'criterias': criterias,'text_models': text_models,"text_datasets": datasets_text,
           "text_modes": text_modes,'defences':defences,'features':feature,'metrics':metric}
    
# archi_text = ['archi1','archi2']

    
@app.route('/train_image', methods=['POST'])
def train_image():
    print(request.form)
    msg = "Training started"
    flash(msg)
    # configs = [os.listdir('msa_toolbox/ui_flask/configs/image'),os.listdir('msa_toolbox/ui_flask/configs/text')]

    return redirect("/training")
    
@app.route('/train_text', methods=['POST'])
def train_text():
    print(request.form)
    msg = "Training started"
    flash(msg)
    thread = Thread(target = main_text(),args=('msa_toolbox/ui_flask/configs/image'+request.form['config_name'],))
    thread.start()
    # configs = [os.listdir('msa_toolbox/ui_flask/configs/image'),os.listdir('msa_toolbox/ui_flask/configs/text')]

    return redirect("/training")
    
@app.route('/file_content')
def get_file_progress():
    with open(path_log) as f:
        content = f.read()
        return content
    
@app.route('/recents',methods=['GET'])
def recents():
    return render_template('recents.html',active = 'recents')

@app.route('/get_stats',methods=['GET'])
def get_stats():
    with open("msa_toolbox/ui_flask/image_runs.json", 'r') as file:
        stats_data = json.load(file)
    print(stats_data)
    return jsonify(stats_data)

@app.route('/process_clicked_row', methods=['GET','POST'])
def process_clicked_row():
    global path_json
    data = request.get_json()
    path_json = data['folder']
    # Your processing logic here
    print(data)
    # Assuming you have a template named 'new_page.html'
    return jsonify({'redirect_url': url_for('progress')})


@app.route('/training', methods=['GET'])
def tranning():
    configs = [os.listdir('msa_toolbox/ui_flask/configs/image'),os.listdir('msa_toolbox/ui_flask/configs/text')]
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
@app.route('/config_text', methods=['GET','POST'])
def submit_text():
    if request.method == 'POST':
        msg = extract_data_text(dict(request.form))
        # print(request.form)
        flash(msg)
        return render_template('index1.html', options=options)
    return render_template('index1.html', options=options)


def extract_data(form):
    return extract_data_image(form)
    

@app.route('/chart',methods=['GET','POST'])
def chart():
    # metric_path = os.path.join(os.getcwd(), 'msa_toolbox/ui_flask/logs/log_metrics.json')
    # log_path = request.args.get('processed_data', default=None)
    # print(log_path)
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
        return render_template('progress.html')
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
    # remove the .yaml extension
    config_list = [x[:-5] for x in os.listdir('msa_toolbox/ui_flask/configs/image')]
    
    return jsonify(config_list)

@app.route('/get_existing_config_files_text', methods=['GET'])
def get_existing_config_files_text():
    # Logic to retrieve a list of existing config file names
    # Return the list as JSON
    # remove the .yaml extension
    config_list = [x[:-5] for x in os.listdir('msa_toolbox/ui_flask/configs/text')]
    
    return jsonify(config_list)

@app.route('/get_config_data', methods=['GET'])
def get_config_data():
    selected_config_file = request.args.get('selected_config_file')
    if selected_config_file == 'create_new':
        # Return an empty structure for a new config file
        return jsonify({'config_data': {}, 'config_file_name': ''})

    # Logic to fetch the data of the selected config file
    selected_config_file_data = yaml.load(open('msa_toolbox/ui_flask/configs/image/'+selected_config_file+'.yaml'), Loader=yaml.FullLoader)
    # Return the data as JSON
    print(selected_config_file)
    return jsonify({'config_data': dict(selected_config_file_data), 'config_file_name': selected_config_file})
@app.route('/get_config_data_text', methods=['GET'])
def get_config_data_text():
    selected_config_file = request.args.get('selected_config_file')
    if selected_config_file == 'create_new':
        # Return an empty structure for a new config file
        return jsonify({'config_data': {}, 'config_file_name': ''})
    else:
        # Logic to fetch the data of the selected config file
        selected_config_file_data = yaml.load(open('msa_toolbox/ui_flask/configs/text/'+selected_config_file+'.yaml'), Loader=yaml.FullLoader)
        # Return the data as JSON
        print(selected_config_file)
        return jsonify({'config_data': dict(selected_config_file_data), 'config_file_name': selected_config_file})

# app.run(host='127.0.0.1', port=8085, debug=True)

app.run(host='0.0.0.0', port=8080, debug=True)
