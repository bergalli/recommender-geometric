FROM python:3.8-slim

# ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qqy update && apt-get install -qqy \
    software-properties-common \
    wget

# Install cuda 11.7
RUN wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-debian11-11-7-local_11.7.0-515.43.04-1_amd64.deb
RUN dpkg -i cuda-repo-debian11-11-7-local_11.7.0-515.43.04-1_amd64.deb
RUN cp /var/cuda-repo-debian11-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/

RUN add-apt-repository contrib && apt-get update
RUN apt-get -y install cuda

# ENV  PATH="/usr/local/cuda-11.7/bin:$PATH"

# # get latest python versions from deadsnakes repository
# RUN add-apt-repository ppa:deadsnakes/ppa
#
# RUN apt-get -qqy update && apt-get install -qqy \
# #         build-essential \
# #         libssl-dev \
# #         libffi-dev \
#         python3.8 \
#         python3.8-dev \
#         python3.8-distutils \
#         python3.8-tk \
#         python3-pip \
#         python3-crcmod

# RUN pip3 install -U pip setuptools requests
RUN pip3 install poetry==1.3.2

ENV PIP_DEFAULT_TIMEOUT=100

COPY poetry.lock  /app/poetry.lock
COPY pyproject.toml  /app/pyproject.toml

WORKDIR /app

# RUN poetry config virtualenvs.create false && poetry install


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
