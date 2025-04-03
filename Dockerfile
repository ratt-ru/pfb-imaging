FROM ubuntu:focal

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
    python -m pip install -U pfb-imaging@git+https://github.com/ratt-ru/pfb-imaging@main && \
    python -m pip cache purge
