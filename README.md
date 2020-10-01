# [YOLO Detector Website] - An Amazing website(https://v-iashin.github.io/detector)

<img src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/typical_russian_day_res.jpeg" alt="Object Detector's Predictions (YOLO v3) on a Sample Image. Caption: 'a man in a business suit and a person in a bear consume are walking on a sidewalk, which is surrounded by cars on a sunny day (Saint Petersburg, Russia)'." width="600">

This repository stores the back-end for the Flask application, which serves requests coming from [the detector at my website](https://v-iashin.github.io/detector).

# How Does it Work
To put it simply, the back-end receives an image from a user and runs an object detection algorithm on the image ([YOLO v3](https://pjreddie.com/darknet/yolo/)). Once the predictions are obtained, they are drawn on the image, which is, then, sent back to the user (to the front-end). In this README I provided the environment setup for the computing machine, which runs the detection algorithm. However, setting up the back-end machine is just the tip of an iceberg. The whole engineering pipeline includes many other steps full of caveats. Specifically, I have undertaken the following steps to build this project:
1. wrote the front-end for the website ([v-iashin/v-iashin.github.io](https://github.com/v-iashin/v-iashin.github.io))
2. reserved a domain name ([Freenom](https://freenom.com/) — I wouldn't recommend it though!)
3. rented an instance and reserved an IP for it ([GoogleCloud](https://cloud.google.com/))
4. added DNS entries mapping my domain to the instance IP ([Freenom](https://freenom.com/) again)
5. signed instance-side digital certificates for HTTPs for my domain ([Let's Encrypt](https://letsencrypt.org/))
6. set up the back-end environment on my instance + detector implementation (THIS repo)

If you are interested in the details of each step, let me know in Issues.

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
