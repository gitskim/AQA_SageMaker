FROM ubuntu:18.04
#FROM pytorch/pytorch:latest
#FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.8.1-cpu-py36-ubuntu18.04



#RUN pip install awscli flask Jinja2 gevent gunicorn

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
         libglib2.0-0 \
         libsm6 \
         libxrender1 \
         libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN pip --no-cache-dir install torch torchvision numpy==1.16.2 scipy==1.2.1 scikit-learn==0.20.2 pandas awscli flask gunicorn boto3==1.9.243 botocore==1.12.243 opencv-python==4.2.0.34
ENV PATH="/opt/program:${PATH}"
COPY getPredScore /opt/program
WORKDIR /opt/program
RUN ls -lh /
RUN chmod +x /opt/program/serve



