FROM ubuntu:focal
MAINTAINER Simon Perkins "lbester@sarao.ac.za"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update && \
    apt -y upgrade && \
    apt install -y \
        git \
        python3-pip \
        python3-minimal \
        python-is-python3 && \
    apt clean


ADD . /src/quartical
RUN python -m pip install -U pip setuptools wheel && \
    python -m pip install -I dask-ms[xarray,s3,arrow,zarr]@git+https://github.com/ratt-ru/dask-ms.git@master  && \
    python -m pip install -U quartical@git+https://github.com/ratt-ru/QuartiCal@master && \
    python -m pip insrall -U pfb-clean@git+https://github.com/ratt-ru/pfb-clean@master && \
    python -m pip install numpy==1.22 && \
    python -m pip cache purge
