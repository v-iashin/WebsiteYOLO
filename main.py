import os
from base64 import b64encode

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from darknet import Darknet
from utils import show_image_w_bboxes_for_server

PROJECT_PATH = './WebsiteYOLO/'

PROJ_TEMP_PATH = os.path.join(PROJECT_PATH, 'proj_tmp')
ARCHIVE_PATH = os.path.join(PROJ_TEMP_PATH, 'upload_archive')
os.makedirs(ARCHIVE_PATH, exist_ok=True)
INPUT_PATH = os.path.join(PROJ_TEMP_PATH, 'input.jpg')
OUTPUT_PATH = os.path.join(PROJ_TEMP_PATH, 'output.jpg')
LOG_PATH = os.path.join(PROJ_TEMP_PATH, 'log.txt')

YOLOV3_WEIGHTS_PATH = os.path.join(PROJECT_PATH, 'weights/yolov3.weights')
YOLOV3_416_CFG_PATH = os.path.join(PROJECT_PATH, 'cfg/yolov3_416x416.cfg')
YOLOV3_608_CFG_PATH = os.path.join(PROJECT_PATH, 'cfg/yolov3_608x608.cfg')
LABELS_PATH = os.path.join(PROJECT_PATH, 'data/coco.names')
FONT_PATH = os.path.join(PROJECT_PATH, 'data/FreeSansBold.ttf')

JPG_QUALITY = 80
DEVICE = torch.device('cpu')

# METHOD = 'yolo_416_coco'
METHOD = 'yolo_608_coco'

# start Flask application
app = Flask(__name__)
# A Flask extension for handling Cross Origin Resource Sharing (CORS),
# making cross-origin AJAX possible.
CORS(app)

if METHOD == 'yolo_608_coco':
    MODEL = Darknet(YOLOV3_608_CFG_PATH)
elif METHOD == 'yolo_416_coco':
    MODEL = Darknet(YOLOV3_416_CFG_PATH)
else:
    raise Exception(f'Undefined method: "{METHOD}"')

MODEL.load_weights(YOLOV3_WEIGHTS_PATH)
MODEL.eval()

assert os.path.exists(PROJECT_PATH), f'{PROJECT_PATH} does not exist. Consider to git clone the repo.'

# if there is no folder for archiving, create
if not os.path.exists(ARCHIVE_PATH):
    os.makedirs(ARCHIVE_PATH)


@app.route('/', methods=['POST'])
def upload_file():
    '''
    Handles a request. If a request to '/' is of the type POST, handle the image
    and add predictions on it.

    Outputs
    -------
    flask.wrappers.Response (POST):
        Outputs the json with filename and image fields (POST).
    '''

    # access files in the request. See the line: 'form_data.append('file', blob);'
    files = request.files['file']
    # save the image ('file') to the disk
    files.save(INPUT_PATH)

    try:
        orientation = request.form['orientation']
        print(f'Submitted orientation: {orientation}')
    except:
        orientation = 'undefined'
        print(vars(request))

    # run the predictions on the saved image
    show_image_w_bboxes_for_server(
        INPUT_PATH, OUTPUT_PATH, ARCHIVE_PATH, LABELS_PATH, FONT_PATH, MODEL, DEVICE, orientation
    )

    # 'show_image_w_bboxes_for_server' saved the output image to the OUTPUT_PATH
    # now we would like to make a byte-file from the save image and sent
    # it back to the user
    with open(OUTPUT_PATH, 'rb') as in_f:
        # so we read an image and decode it into utf-8 string and append it
        # to data:image/jpeg;base64 and then return it.
        img_b64 = b64encode(in_f.read()).decode('utf-8')
        img_b64 = 'data:image/jpeg;base64, ' + img_b64

    return jsonify(name='input.jpg', image=str(img_b64))

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
