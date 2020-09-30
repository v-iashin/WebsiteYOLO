# [A YOLO DETECTOR WEBSITE](https://v-iashin.github.io/detector)

<img src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/typical_russian_day_res.jpeg" alt="Object Detector's Predictions (YOLO v3) on a Sample Image. Caption: 'a man in a business suit and a person in a bear consume are walking on a sidewalk, which is surrounded by cars on a sunny day (Saint Petersburg, Russia)'." width="600">

The repository stores the back-end for the Flask application, which serves the requests coming from [the detector at my website](https://v-iashin.github.io/detector). 

# How Does it Work
To put it simply, it receives a user image and runs an object detection algorithm on it ([YOLO v3](https://pjreddie.com/darknet/yolo/)). Once the predictions are retrieved, the resulting image with bounding boxes is sent back to the front-end. In this README I provide the environment setup for the compute machine. However, setting up the bach-end machine is just the tip of an iceberg. The whole engineering pipeline includes many other steps full of caveats. Just to scratch the surface, I undertook the following steps to build this project: 
1. writen the website front-end ([v-iashin/v-iashin.github.io](https://github.com/v-iashin/v-iashin.github.io))
2. obtained a domain name ([Freenom](https://freenom.com/) — I wouldn't recommend it though!)
3. rented an instance and reserved an IP for it ([GoogleCloud](https://cloud.google.com/))
4. added DNS entries mapping my domain to the instance IP ([Freenom](https://freenom.com/) again)
5. signed instance-side digital certificates for HTTPs for my domain ([Let's Encrypt](https://letsencrypt.org/))
6. setted up the back-end environment on my instance + detector implementation (THIS repo)

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

