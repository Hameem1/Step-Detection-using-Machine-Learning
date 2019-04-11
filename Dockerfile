FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y python3-pip python3-dev
RUN apt-get install -y language-pack-de
RUN cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip

RUN  mkdir /StepDetectionProject
WORKDIR /StepDetectionProject
COPY ./requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 5000 5001
