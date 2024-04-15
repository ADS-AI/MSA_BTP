import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from ...utils.image.cfg_reader import CfgNode
from ...utils.image.load_data_and_models import get_data_loader


'''
Function to be used when either:
Victim has no defense mechansim to use OR model stealing attack is not detected by the defense mechanism 
'''
def label_samples_with_no_defence_clarafai(cfg:CfgNode, thief_data:Dataset, 
            next_training_samples_indices:np.array, take_action:bool=False):
    '''
    Labels the new thief training samples using the victim model
    '''
    # addendum_loader = get_data_loader(Subset(thief_data, next_training_samples_indices), 
                        # batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    addendum_loader = DataLoader(Subset(thief_data, next_training_samples_indices), batch_size=1, shuffle=False)
    
    for img, label0, index in addendum_loader:
        for ii, jj in enumerate(index):
            image_path = thief_data.samples[jj][0]
            if cfg.TRAIN.BLACKBOX_TRAINING == True:
                label = call_victim_api_for_label(cfg, image_path, blackbox=True)
                thief_data.samples[jj] = (thief_data.samples[jj][0], label)
            else:
                label = call_victim_api_for_label(cfg, image_path, blackbox=False)
                thief_data.samples[jj] = (thief_data.samples[jj][0], label)
            print(label)
    print('New labels for the thief data have been obtained')
    return


def call_victim_api_for_label(cfg:CfgNode, image_path, blackbox:bool=True):
    '''
    Calls the victim API to get the label for the image
    '''
    # Your PAT (Personal Access Token) can be found in the portal under Authentification
    PAT = cfg.VICTIM.PERSONAL_ACCESS_TOKEN
    # Specify the correct user_id/app_id     pairings
    # Since you're making inferences outside your app's scope
    USER_ID = cfg.VICTIM.USER_ID
    APP_ID = cfg.VICTIM.APP_ID
    # Change these to whatever model and image URL you want to use
    MODEL_ID = cfg.VICTIM.MODEL_ID
    MODEL_VERSION_ID = cfg.VICTIM.MODEL_VERSION_ID


    ############################################################################
    # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE TO RUN THIS EXAMPLE
    ############################################################################
    from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
    from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
    from clarifai_grpc.grpc.api.status import status_code_pb2

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (('authorization', 'Key ' + PAT),)
    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)
    
    with open(image_path, "rb") as f:
        file_bytes = f.read()
        
    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
            inputs=[
                resources_pb2.Input(
                    id = '1',
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            # url=IMAGE_URL
                            base64=file_bytes,
                            allow_duplicate_url=True
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

    class_to_idx = {}
    probabilities = []
    
    # Since we have one input, one output will exist here
    for i in range(len(post_model_outputs_response.outputs)):
        output = post_model_outputs_response.outputs[i]
        # print("Predicted concepts for input %d" % i)
        # print(output.data.concepts)
        for concept in output.data.concepts:
            class_to_idx[concept.name] = len(class_to_idx)
            probabilities.append(concept.value)
            # print("%s %.2f" % (concept.name, concept.value))
    
    cfg.VICTIM.CLASS_TO_IDX = class_to_idx
    cfg.VICTIM.IDX_TO_CLASS = {v: k for k, v in class_to_idx.items()}
    cfg.VICTIM.NUM_CLASSES = len(class_to_idx)
    if blackbox == True:
        return torch.argmax(torch.tensor(probabilities)).item()
    else:
        return probabilities