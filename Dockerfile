FROM ubuntu:20.04

RUN apt-get -qqy update && apt-get install -qqy software-properties-common
# to get latest python versions
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get -qqy update && apt-get install -qqy \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3.9 \
        python3.9-dev \
        python3-pip \
        python3-crcmod

#RUN pip3 install -U pip setuptools requests
RUN pip3 install poetry

COPY poetry.lock  /app/poetry.lock
COPY pyproject.toml  /app/pyproject.toml

#ARG CI_JOB_TOKEN
#ARG CI_JOB_TOKEN_NAME

WORKDIR /app

RUN poetry config virtualenvs.create false && poetry install
#RUN poetry config http-basic.aeye $CI_JOB_TOKEN_NAME $CI_JOB_TOKEN \
#    && poetry config repositories.aeye https://gitlab.com/api/v4/projects/24685092/packages/pypi/simple
#RUN poetry install --no-dev

COPY conf /app/conf
COPY main /app/main
COPY src /app/src

RUN mkdir /app/logs && touch info.log

ENV PYTHONPATH=./:/app/src:/app/main:$PYTHONPATH
ARG GOOGLE_APPLICATION_CREDENTIALS
ENV GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS

ENTRYPOINT ["python3", "main/local_cli.py"]