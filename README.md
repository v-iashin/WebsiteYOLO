# [YOLO Detector Website](https://v-iashin.github.io/detector)

<img src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/typical_russian_day_res.jpeg" alt="Object Detector's Predictions (YOLO v3) on a Sample Image. Caption: 'a man in a business suit and a person in a bear consume are walking on a sidewalk, which is surrounded by cars on a sunny day (Saint Petersburg, Russia)'." width="600">

This repository stores the back-end for the web app object detector, which serves requests coming from
[this web page](https://v-iashin.github.io/detector).

# How Does it Work
To put it simply, the back-end receives an image from a user and runs an object detection algorithm on the image ([YOLO v3](https://pjreddie.com/darknet/yolo/)).
Once the predictions are obtained, they are drawn on the image, which is, then, sent back to the user (to the front-end).

One could implement the back-end as a **Flask app** on or as **Gradio app** with their generic front-end (later is currently used).

With HuggingFace's Spaces, it is possible to easily deploy the Gradio UI and embed their interface into
the html of the website you want.
Luckily, the Spaces provide a small free-tier CPU machine which is enough to serve this detector (perhaps a bit slow at times though).

If you want to deploy the detector on your premises as a Gradio or Flask app, it will require more work but
this approach is more flexible and allows you to customize the front-end and compute power as you wish.
Doing so, however, includes many steps full of caveats.
If you are interested in the details of each step on how to do it with Flask,
checkout [How Did You Build Your Object Detector?](https://v-iashin.github.io/how_did_you_build_your_detector).
In this tutorial, you will find the details on how to write the front- and back-end, how to rent a domain name,
setup DNS entries, SSL certificates, and deploy the app on a remote machine.

In this README I provided the environment setup for both Gradio and Flask.

# Setting up the Environment

You can use either conda or docker to setup the environment which supports both Flask and Gradio.

## conda
Download the YOLOv3 weights
```bash
bash ./weights/download_weights_yolov3.sh
```

Install the [conda](https://docs.conda.io/en/latest/miniconda.html) environment
```bash
conda env create -f ./conda_env.yml
conda activate detector
```

## docker
Assuming that docker is installed, start by building the docker image using the
provided `Dockerfile`
```bash
docker build - < Dockerfile --tag website_yolo
```

# Running the detector

Once the environment is setup, you can run the detector as a Flask or Gradio app.
You may want to run the Flask server under a `tmux` session.

## Gradio

```bash
conda activate detector
python ./gradio_app.py
```
Open in browser as `http://127.0.0.1:7860`

## Flask

### conda
```bash
conda activate detector
export FLASK_APP=./WebsiteYOLO/main.py
# export FLASK_RUN_CERT=/etc/letsencrypt/live/your.domain/fullchain.pem
# export FLASK_RUN_KEY=/etc/letsencrypt/live/your.domain/privkey.pem
flask run --host=0.0.0.0
```
now you can send `POST` requests to `http://127.0.0.1:5000/` (root) and `GET` requests
to `http://127.0.0.1:5000/status_check` (openable in the browser).

### docker
Once the docker image `website_yolo` is built, run
```bash
docker run \
    -p 5000:5000 \
    -v /etc/letsencrypt:/etc/letsencrypt \
    -v /home/ubuntu/WebsiteYOLO/proj_tmp:/home/user/app/WebsiteYOLO/proj_tmp \
    website_yolo
```
where `5000` is the default port which flask uses and docker exposes.
You mount the `/etc/letsencrypt` from the host where you store your SSL certificates
to `/etc/letsencrypt` folder in the docker container.
Also, this will mount `/home/ubuntu/WebsiteYOLO/proj_tmp` (make sure it exists)
to gather the submitted images.
