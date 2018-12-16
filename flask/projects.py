from flask import Flask, request, jsonify
from flask_cors import CORS
from time import time
from base64 import b64encode
import os
import dlib
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

app = Flask(__name__)
CORS(app)

ARCHIVE_PATH = 'upload_archive/'
OUTPUT_PATH = 'project/flask/output.jpg'
INPUT_PATH = 'project/flask/input.jpg'

def show_image_w_bboxes_for_server(img_path, method):
    img = dlib.load_rgb_image(img_path)
    plt.figure(figsize=(7, 7))
    
    if method == 'dlib_cnn_1':
        weights = 'weights/dlib_cnn_detector_weights.dat'
        detector = dlib.cnn_face_detection_model_v1(weights)
        faces = detector(img, 1)
        
        for face in faces:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
    elif method == 'opencv_haar':
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        weights_path = 'project/weights/haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(weights_path)
        faces = detector.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
        
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    elif method == 'dlib_hog_1':
        detector = dlib.get_frontal_face_detector()
        faces = detector(img, 1)

        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    else:
        raise Exception('Undefined method: "{}"'.format(method))
    
    dlib.save_image(img, OUTPUT_PATH)
    dlib.save_image(img, ARCHIVE_PATH + str(time()) + '.jpg')


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        files = request.files['file']
        files.save(INPUT_PATH)
        file_size = os.path.getsize(INPUT_PATH)
        show_image_w_bboxes_for_server(INPUT_PATH, 'opencv_haar')

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