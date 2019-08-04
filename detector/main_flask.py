from flask import Flask, request, jsonify
from flask_cors import CORS
from time import time
from base64 import b64encode
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
from matplotlib import pyplot as plt
from utils import parse_cfg, predict_and_save
from darknet import Darknet

###
PROJECT_PATH = './PersonalProjects/'

PROJ_TEMP_PATH = os.path.join(PROJECT_PATH, 'proj_tmp')
ARCHIVE_PATH = os.path.join(PROJ_TEMP_PATH, 'upload_archive')
INPUT_PATH = os.path.join(PROJ_TEMP_PATH, 'input.jpg')
OUTPUT_PATH = os.path.join(PROJ_TEMP_PATH, 'output.jpg')
LOG_PATH = os.path.join(PROJ_TEMP_PATH, 'log.txt')

DETECTOR_PATH = os.path.join(PROJECT_PATH, 'detector')
YOLOV3_WEIGHTS_PATH = os.path.join(DETECTOR_PATH, 'weights/yolov3.weights')
YOLOV3_416_CFG_PATH = os.path.join(DETECTOR_PATH, 'cfg/yolov3_416x416.cfg')
YOLOV3_608_CFG_PATH = os.path.join(DETECTOR_PATH, 'cfg/yolov3_608x608.cfg')
YOLOV3_LABELS_PATH = os.path.join(DETECTOR_PATH, 'data/coco.names')
FONT_PATH = os.path.join(DETECTOR_PATH, 'data/FreeSansBold.ttf')

JPG_QUALITY = 80
DEVICE = torch.device('cpu:0')

METHOD = 'yolo_608_coco'
###

# start Flask application
app = Flask(__name__)
CORS(app)

if METHOD is 'yolo_608_coco':
    MODEL = Darknet(YOLOV3_608_CFG_PATH)

elif METHOD is 'yolo_416_coco':
    MODEL = Darknet(YOLOV3_416_CFG_PATH)
    
else:
    raise Exception(f'Undefined method: "{METHOD}"')
    
MODEL.load_weights(YOLOV3_WEIGHTS_PATH)
MODEL.eval()

assert os.path.exists(PROJECT_PATH), f'{PROJECT_PATH} does not exist. Consider to git clone the repo.'
    
# if there is no folder for archiving, create
if not os.path.exists(ARCHIVE_PATH):
    os.makedirs(ARCHIVE_PATH)
    
def show_image_w_bboxes_for_server(img_path, model, orientation):
    '''
    Reads an image from the disk and applies a detection algorithm specified in model.
    
    Arguments
    ---------
    img_path: str
        A path to an image.
    model: Darknet
        Model to apply to the image.
    orientation: str
        Orientation which front-end tries to extract from EXIF of an image.
        Can be 'undefined' or some integer which can be used to orient the image.
        Used in predict_and_save().
    '''
    
    # I want to log the processing time for each image
    start = time()
    
    # predict_and_save returns both img with predictions drawn on it 
    # and the tensor with predictions
        
    with torch.no_grad():
        # TODO: add captital-letter arguments to the argument list
        predictions, img = predict_and_save(
            img_path, OUTPUT_PATH, model, DEVICE, YOLOV3_LABELS_PATH, 
            FONT_PATH, orientation, show=False
        )
        
    # selecting a name for a file for archiving
    filename = f'{time()}.jpg'
    archive_full_path = os.path.join(ARCHIVE_PATH, filename)
    img.save(archive_full_path, 'JPEG')
    img.save(OUTPUT_PATH, 'JPEG')
    
    # calculating elapsed time and printing it to flask console
    elapsed_time = round(time() - start, 2)
    
    print(f'Processing time of {filename}: {elapsed_time} sec.')
    print('=' * 70)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    '''
    Handles a request. If a request to '/' is of the type POST, hangle the image 
    and add predictions on it; if a request of the type GET, return a notification
    that GET request has been received.
    
    Outputs
    -------
    flask.wrappers.Response (POST), str (GET):
        Outputs the json with filename and image fields (POST); returns a string 
        if a GET request is sent.
    '''

    if request.method == 'POST':
        # access files in the request. See the line: 'form_data.append('file', blob);'
        files = request.files['file']
        try:
            orientation = request.form['orientation']
            print(orientation)
            
#             # which means that there is no EXIF in the user's image
#             if orientation is 'undefined':
#                 orientation = -1
            
#             # front-end's FormData sends the info in strings
#             orientation = int(orientation)
            
        except:
            orientation = 'undefined'
            print(vars(request))
            
        # save the image ('file') to the disk
        files.save(INPUT_PATH)
        # run the predictions on the saved image
        show_image_w_bboxes_for_server(INPUT_PATH, MODEL, orientation)
        
        # 'show_image_w_bboxes_for_server' saved the output image to the OUTPUT_PATH
        # now we would like to make a byte-file from the save image and sent
        # it back to the user
        with open(OUTPUT_PATH, 'rb') as in_f:
            # so we read an image and decode it into utf-8 string and append it 
            # to data:image/jpeg;base64 and then return it.
            img_b64 = b64encode(in_f.read()).decode('utf-8')
            img_b64 = 'data:image/jpeg;base64, ' + img_b64
    
        return jsonify(name='input.jpg', image=str(img_b64))

    elif request.method == 'GET':
        return 'GET request received'
    
@app.route('/status_check', methods=['GET'])
def status_check():
    '''
    This endpoint for the status check indicator on the site.
    
    Output
    ------
    str:
        Returns a string if a GET request is sent.
    '''

    if request.method == 'GET':
        return 'GET request received'