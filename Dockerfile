FROM ubuntu:18.04 
ARG workspace=/late-residual

RUN mkdir ${workspace}
COPY requirements.txt ${workspace}/requirements.txt
COPY Code ${workspace}/Code
COPY CleaningScripts ${workspace}/CleaningScripts

# Install python 3.8
RUN apt-get update && apt-get install -y python3.8 && apt-get install -y python3-pip
# alias python3.8 for all python commands
RUN ln -s /usr/bin/python3.8 /usr/bin/python

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install -r ${workspace}/requirements.txt

