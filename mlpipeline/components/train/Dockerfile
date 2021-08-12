FROM ubuntu:20.04
# Install some common dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qqy && \
    apt-get install -y \
      build-essential \
      cmake \
      g++ \
      git \
      libsm6 \
      libxrender1 \
      locales \
      libssl-dev \
      pkg-config \
      poppler-utils \
      python3-dev \
      python3-pip \
      python3.6 \
      software-properties-common \
      && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Set default python version to 3.6
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip3 install -U pip setuptools

# locale setting
RUN locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    PYTHONIOENCODING=utf-8

RUN pip install torch numpy comet_ml boto3 requests opencv-python
RUN pip install flask
RUN apt update && apt install ffmpeg libsm6 libxext6  -y
COPY ./src /src