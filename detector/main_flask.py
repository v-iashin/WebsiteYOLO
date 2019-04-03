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

PROJECT_PATH = './PersonalProjects/'

PROJ_TEMP_PATH = os.path.join(PROJECT_PATH, 'proj_tmp')
ARCHIVE_PATH = os.path.join(PROJ_TEMP_PATH, 'upload_archive')
INPUT_PATH = os.path.join(PROJ_TEMP_PATH, 'input.jpg')
OUTPUT_PATH = os.path.join(PROJ_TEMP_PATH, 'output.jpg')

DETECTOR_PATH = os.path.join(PROJECT_PATH, 'detector')
HAAR_WEIGHTS_PATH = os.path.join(DETECTOR_PATH, 'weights/haarcascade_frontalface_default.xml')
YOLOV3_WEIGHTS_PATH = os.path.join(DETECTOR_PATH, 'weights/yolov3.weights')
YOLOV3_416_CFG_PATH = os.path.join(DETECTOR_PATH, 'cfg/yolov3_416x416.cfg')
YOLOV3_608_CFG_PATH = os.path.join(DETECTOR_PATH, 'cfg/yolov3_608x608.cfg')
YOLOV3_LABELS_PATH = os.path.join(DETECTOR_PATH, 'data/coco.names')

JPG_QUALITY = 80
DEVICE = torch.device('cpu:0')

# todo if with methods so it will take less memory
DARKNET_416 = Darknet(YOLOV3_416_CFG_PATH)
DARKNET_416.load_weights(YOLOV3_WEIGHTS_PATH)
DARKNET_416.eval();
DARKNET_608 = Darknet(YOLOV3_608_CFG_PATH)
DARKNET_608.load_weights(YOLOV3_WEIGHTS_PATH)
DARKNET_608.eval();
HAAR_DETECTOR = cv2.CascadeClassifier(HAAR_WEIGHTS_PATH)

METHOD = 'yolo_608_coco'

app = Flask(__name__)
CORS(app)

assert os.path.exists(PROJECT_PATH), f'{PROJECT_PATH} does not exist. Consider to git clone the repo.'
    
if not os.path.exists(ARCHIVE_PATH):
    os.makedirs(ARCHIVE_PATH)
    
def show_image_w_bboxes_for_server(img_path, method):
    
    start = time()
    
    if method == 'yolo_416_coco':
        
        with torch.no_grad():
            _, img = predict_and_save(img_path, OUTPUT_PATH, DARKNET_416, 
                                      DEVICE, YOLOV3_LABELS_PATH, show=False)
        
    if method == 'yolo_608_coco':
        
        with torch.no_grad():
            _, img = predict_and_save(img_path, OUTPUT_PATH, DARKNET_608, 
                                      DEVICE, YOLOV3_LABELS_PATH, show=False)
        
    
    elif method == 'opencv_haar':
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = HAAR_DETECTOR.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
        
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
    else:
        raise Exception('Undefined method: "{}"'.format(method))
    
    filename = f'{time()}.jpg'
    archive_full_path = os.path.join(ARCHIVE_PATH, filename)
    cv2.imwrite(archive_full_path, img, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
    cv2.imwrite(OUTPUT_PATH, img, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
    
    print(f'Processing time of {filename}: {round(time() - start, 2)} sec.')
    print('=' * 50)

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        method = METHOD
        files = request.files['file']
        files.save(INPUT_PATH)
        file_size = os.path.getsize(INPUT_PATH)
#         show_image_w_bboxes_for_server(INPUT_PATH, 'opencv_haar')
        show_image_w_bboxes_for_server(INPUT_PATH, method)

        with open(OUTPUT_PATH, 'rb') as in_f:
            img_b64 = b64encode(in_f.read()).decode('utf-8')
            img_b64 = 'data:image/jpeg;base64, ' + img_b64

        return jsonify(name='input.jpg', size=file_size, image=str(img_b64))

    elif request.method == 'GET':
        return 'GET request received'
    
@app.route('/status_check', methods=['GET'])
def status_check():

    if request.method == 'GET':
        return 'GET request received'