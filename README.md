# [Website YOLO Detector](https://v-iashin.github.io/detector)

The repository stores the back-end for the Flask application, which serves the requests coming from [the detector at my website](https://v-iashin.github.io/detector). To put it simply, it receives a user image and runs a object detection algorithm ([YOLO v3](https://pjreddie.com/darknet/yolo/)). Once the predictions are retrieved, the resulting image with bounding boxes is sent back to the front-end.

In this README I provide the environment setup. However, this is just the tip of an iceberg. The whole engineering pipeline includes many other things full of caveats. 

Just to scratch the surface, I undertook the following steps to build this project: 
1. writen the website front-end ([v-iashin/v-iashin.github.io](https://github.com/v-iashin/v-iashin.github.io))
2. obtained a domain name ([Freenom](freenom.com) -- I wouldn't recommend it though!)
3. rented an instance and reserved an IP for it ([GoogleCloud](https://cloud.google.com/))
4. added DNS entries mapping my domain to the instance IP ([Freenom again](freenom.com) -- I wouldn't recommend it though!)
5. signed instance-side digital certificates for HTTPs for my domain ([Let's Encrypt](https://letsencrypt.org/))
6. setted up the back-end environment on my instance (THIS)

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

