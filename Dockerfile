# Pull a base image.
FROM ubuntu:18.04

# Install libraries in the brand new image. 
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         python3-dev \
         build-essential \
         nginx \
         ca-certificates \
         libglib2.0-0 \
         libsm6 \
         libxrender1 \
         libxext6 \
         zlib1g-dev \
         libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 3 setup
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Set variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN pip --no-cache-dir install torch torchvision numpy==1.16.2 scipy==1.2.1 scikit-learn==0.20.2 pandas awscli flask gunicorn boto3==1.9.243 botocore==1.12.243 opencv-python==4.2.0.34

# Set an executable path
ENV PATH="/opt/program:${PATH}"

# Copy my SageMaker inference code package to the executable path
COPY getPredScore /opt/program

# Set the working directory for all the subsequent Dockerfile instructions.
WORKDIR /opt/program

# Make the serve file executable
RUN chmod +x /opt/program/serve



