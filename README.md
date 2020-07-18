# Website YOLO Detector

The repository stores the back-end for the Flask application, which serves the requests coming from [the detector at my website](https://v-iashin.github.io/detector). The received user images are processed by [YOLO v3](https://pjreddie.com/darknet/yolo/). Once done, the resulting image with bounding boxes are sent back to the front-end.

This setup guide is covering the environment setup for the Flask application. However, this is just the tip of an iceberg. The whole engineering pipeline includes many other things full of caveats. Just to scratch the surface, you will need to undertake the following steps: 1) write the website front-end; 2) obtain a domain name; 3) rent an instance and reserve an IP for it; 4) add DNS entries mapping your domain to the instance IP; 5) sign instance-side digital certificates for HTTPs for your domain; 6) Setup the back-end environment on your instance (THIS).

If you are interested to know how the rest of the project is organized, let me know in Issues. 

# Setup

Download the YOLOv3 weights
```bash
bash ./weights/download_weights_yolov3.sh
```

Install the [conda](https://docs.conda.io/en/latest/miniconda.html) environment
```bash
conda env create -f ./conda_env.yml
conda activate detector
```

