FROM nvidia/cuda:12.0.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_VERSION=1.3.2

RUN apt-get -qqy update && apt-get install -qqy \
    software-properties-common \
    wget \
    curl

# Install nvidia tesla drivers
# RUN BASE_URL=https://us.download.nvidia.com/tesla && \
#     DRIVER_VERSION=450.80.02 && \
#     curl -fSsl -O $BASE_URL/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run \
#
# RUN apt-get install intel-microcode && sh NVIDIA-Linux-x86_64-$DRIVER_VERSION.run

# RUN add-apt-repository contrib non-free
# RUN apt-get -qqy install nvidia-driver


# get latest python versions from deadsnakes repository
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -qqy update && apt-get install -qqy \
#         build-essential \
#         libssl-dev \
#         libffi-dev \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        python3.8-tk \
        python3-pip \
        python3-crcmod

# RUN pip3 install -U pip setuptools requests
RUN pip3 install poetry==$POETRY_VERSION

# to bypass timeout error on long download time
ENV PIP_DEFAULT_TIMEOUT=10000

COPY poetry.lock  /app/poetry.lock
COPY pyproject.toml  /app/pyproject.toml

WORKDIR /app

RUN poetry install --without dev

# RUN poetry config virtualenvs.create false # this line raise an error with nvidia cuda base image


#
# COPY conf /app/conf
# COPY extras /app/extras
# COPY src /app/src
# COPY cli.py /app/cli.py
#
# ENV PYTHONPATH=./:/app/src:/app/main:$PYTHONPATH
# ARG GOOGLE_APPLICATION_CREDENTIALS
# ENV GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS
#
# ENTRYPOINT ["python3", "cli.py"]
