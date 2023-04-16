import numpy as np
import gradio as gr
import yaml
import os
import time
from threading import Thread
# from msa_toolbox.main import app
from .. main import app

config_dir = './msa_toolbox/ui_gradio/configs/'
lis = os.listdir(config_dir)


def count():
    while True:
        s = ''
        with open("/home/vikram/akshitj/toolbox_btp/main_toolbox/MSA_BTP/Utils/Logs/log.txt", 'r') as f:
            s = f.read()
        yield s


def start_training(config_name):
    config_path = config_dir+config_name
    print(config_path, config_dir)
    print(os.listdir(config_dir))
    if config_name in os.listdir(config_dir):
        thread = Thread(target=app, args=(config_path,))
        thread.start()
        return 'Model Trained'
    else:
        return 'config file not found'
    # return config_path


def make_cfg(name, t_dataset, v_dataset, subset, t_model, v_model, Budget, Method, Optimizer, Criteria, Device, Output):
    victim = {'DATASET': v_dataset,
              'ARCHITECTURE': v_model,
              'WEIGHTS': 'default'}

    thief = {'DATASET': t_dataset,
             'ARCHITECTURE': t_model,
             'SUBSET': subset,
             'WEIGHTS': 'default'}

    active = {'METHOD': Method,
              'BUDGET': Budget}
    train = {'OPTIMIZER': Optimizer,
             'CRITERIA': Criteria}

    cfg = {'VICTIM': victim,
           'THIEF': thief,
           'ACTIVE': active,
           'TRAIN': train,
           'DEVICE': Device,
           'OUTPUT': Output}
    # print(cfg)

    yaml_string = yaml.dump(cfg, default_flow_style=False)
    print("The YAML string is:")
    print(yaml_string)

    # save the yaml file to the disk
    path = config_dir+name+'.yaml'
    yaml.dump(cfg, open(path, 'w'))

    return 'config file generated'


def x():
    lis = os.listdir(config_dir)
    ret = ''
    for i in lis:
        ret += i+'\n'
    return ret


with gr.Blocks() as demo:
    gr.Markdown("G_gradio for MSA-TOOLBOX")
    with gr.Tab("Generate Config"):
        input = [
            gr.Textbox(lines=1, label="Config Name",
                       info="Enter the Config Name"),

            gr.Dropdown(
                ['cifar10', 'cifar100', 'imagenet', 'mnist', 'kmnist', 'fashionmnist', 'emnist', 'emnistletters', 'svhn', 'tinyimagenet200', 'tinyimagesubset', 'cubs200',
                 'diabetic5', 'indoor67', 'caltech256'],
                label="Thief Dataset",
                info="Choose the Thief Dataset"
            ),
            gr.Dropdown(
                ['cifar10', 'cifar100', 'imagenet', 'mnist', 'kmnist', 'fashionmnist', 'emnist', 'emnistletters', 'svhn', 'tinyimagenet200', 'tinyimagesubset', 'cubs200',
                 'diabetic5', 'indoor67', 'caltech256'],
                label="Victim Dataset",
                info="Choose the Thief Dataset"
            ),

            gr.Textbox(lines=1, label="Subset",
                       info="Choose the Subset size of Thief dataset"
                       ),
            gr.Dropdown(
                ['resnet18', 'resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'alexnet', 'efficientnet_b0', 'efficientnet_b1',
                 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
                 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small'], label="Victim Model",
                info="Choose the Victim Architecture"
            ),
            gr.Dropdown(
                ['resnet18', 'resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'alexnet', 'efficientnet_b0', 'efficientnet_b1',
                 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
                 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small'], label="Thief Model",
                info="Choose the Victim Architecture"
            ),
            gr.Textbox(lines=1, label="Budget",
                       info="Enter the Budget in RS"
                       ),
            gr.Dropdown(
                ['entropy'],
                label="Method",
                info="Choose the Method for Active learning"
            ),
            gr.Dropdown(
                ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax'],
                label="Optimizer",
                info="Choose the Optimizer"
            ),
            gr.Dropdown(
                ['cross_entropy_loss', 'mse_loss', 'l1_loss',
                    'soft_margin_loss', 'bce_loss'],
                label="Creteria",
                info="Choose the Creteria for Optimizer"
            ),
            gr.Dropdown(
                ['cpu', 'gpu'],
                label="Device",
                info="Choose the Device for Training"
            ),
            gr.Textbox(lines=1, label="Output",
                       info="Enter the Output Directory"
                       ),
        ],
        text_button = gr.Button("Generate Config"),
        text_output = gr.Textbox()
    with gr.Tab("Start Tranning"):
        with gr.Column():
            output = gr.Textbox(lines=15, label="Configs",
                                info="Available Configs")
            config_button = gr.Button("Update Configs")

        with gr.Column():
            inputs = gr.Textbox(lines=1, label="Config",
                                info="Name of the Config file")
            output_tran = gr.Textbox(
            lines=15, label="Progress", info="Progress of the Training")
            train_button = gr.Button("Start Training")
    # with gr.Tab("View Results"):
    #     prog_out = gr.Textbox(label="Output Box")
    #     prog_button = gr.Button("View Progress")
    # with gr.Accordion("Open for More!"):
    #     gr.Markdown("Look at me...")
    # import pdb;pdb.set_trace()
    text_button[0].click(make_cfg, inputs=input[0], outputs=text_output)
    config_button.click(x, inputs=None, outputs=output)
    train_button.click(start_training, inputs=inputs, outputs=output_tran)
    # prog_button.click(count, inputs=None, outputs=prog_out)


demo1 = gr.Interface(count, inputs=None,
                     outputs=gr.Textbox(label="Output Box",max_lines=1000))

demo2 = gr.TabbedInterface([demo, demo1], ['Setup Config', 'View Progress'])
demo2.queue()
demo2.launch()
