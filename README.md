# [YOLO Detector Website](https://v-iashin.github.io/detector)

<img src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/typical_russian_day_res.jpeg" alt="Object Detector's Predictions (YOLO v3) on a Sample Image. Caption: 'a man in a business suit and a person in a bear consume are walking on a sidewalk, which is surrounded by cars on a sunny day (Saint Petersburg, Russia)'." width="600">

This repository stores the back-end for the Flask application, which serves requests coming from [the detector at my website](https://v-iashin.github.io/detector).

# How Does it Work
To put it simply, the back-end receives an image from a user and runs an object detection algorithm on the image ([YOLO v3](https://pjreddie.com/darknet/yolo/)).
Once the predictions are obtained, they are drawn on the image, which is, then, sent back to the user (to the front-end).
In this README I provided the environment setup for the computing machine, which runs the detection algorithm.
However, setting up the back-end machine is just the tip of an iceberg.
The whole engineering pipeline includes many other steps full of caveats.
If you are interested in the details of each step, checkout [How Did You Build Your Object Detector?](https://v-iashin.github.io/how_did_you_build_my_detector).

# Setting up the Environment
Download the YOLOv3 weights
```bash
bash ./weights/download_weights_yolov3.sh
```

Install the [conda](https://docs.conda.io/en/latest/miniconda.html) environment
```bash
conda env create -f ./conda_env.yml
conda activate detector
```

# Running the detector as a Flask app:
```bash
conda activate detector
export FLASK_APP=./WebsiteYOLO/main.py
# export FLASK_RUN_CERT=/etc/letsencrypt/live/your.domain/fullchain.pem
# export FLASK_RUN_KEY=/etc/letsencrypt/live/your.domain/privkey.pem
flask run --host=0.0.0.0
```
