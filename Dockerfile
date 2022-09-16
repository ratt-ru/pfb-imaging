FROM ubuntu:focal
MAINTAINER Landman Bester "lbester@sarao.ac.za"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update && \
    apt -y upgrade && \
    apt install -y \
        git \
        python3-pip \
        python3-minimal \
        python-is-python3 && \
    apt clean


RUN python -m pip install -U pip setuptools wheel && \
    python -m pip install -U pfb-clean@git+https://github.com/ratt-ru/pfb-clean@awskube && \
    python -m pip install numpy==1.22 && \
    python -m pip cache purge
