FROM python:3.6.8-stretch

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /stage/

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential \
    glpk-utils \
    libglpk-dev \
    openjdk-8-jdk \
    gettext-base && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install nltk
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader punkt
RUN python3 -m pip install --no-cache-dir \
    mkl \
    torch \
    tensorboard


COPY README.md .
COPY setup.py .
COPY transformers-cli .
COPY src/transformers src/transformers
RUN python3 -m pip install --no-cache-dir .
COPY src/modularqa src/modularqa
COPY examples/ examples/





LABEL maintainer="tushark@allenai.org"

CMD ["/bin/bash"]
