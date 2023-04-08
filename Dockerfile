FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y sudo git curl wget

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
# COPY --chown=user . $HOME/app

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="/home/user/miniconda3/bin:$PATH"
RUN echo ". /home/user/miniconda3/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init bash
RUN conda config --set auto_activate_base false

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Clone the repository
RUN git clone https://github.com/v-iashin/WebsiteYOLO.git

# Set the working directory to WebsiteYOLO
WORKDIR $HOME/app/WebsiteYOLO

# download the weights
RUN bash ./weights/download_weights_yolov3.sh

# Installing conda environment
RUN conda env create -f conda_env.yml
RUN conda clean -afy
RUN rm ../../Miniconda3-latest-Linux-x86_64.sh

# going out of the dir
WORKDIR $HOME/app

ENV FLASK_APP=WebsiteYOLO/main.py
ENV FLASK_RUN_PORT=5000
ENV FLASK_RUN_CERT=/etc/letsencrypt/live/detector.iashin.ai/fullchain.pem
ENV FLASK_RUN_KEY=/etc/letsencrypt/live/detector.iashin.ai/privkey.pem

# start the server
CMD [ "conda", "run", "-n", "detector", "flask", "run", "--host", "0.0.0.0" ]

# to run use `docker run -p 5000:5000 -v /etc/letsencrypt:/etc/letsencrypt -v /home/ubuntu/WebsiteYOLO/proj_tmp:/home/user/app/WebsiteYOLO/proj_tmp website_yolo`
# make sure `/home/ubuntu/WebsiteYOLO/proj_tmp` exists on the host machine
