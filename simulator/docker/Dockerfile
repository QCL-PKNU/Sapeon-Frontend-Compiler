FROM ubuntu:18.04
LABEL org.opencontainers.image.authors="brown@sapeon.com"
USER root
ENV PB_VERSION=3.16.0
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/bin:${PATH}
SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get -y install gcc-9 g++-9 gdb git vim

RUN apt-get -y install wget python3.8 python3.8-dev python3-pip libgtest-dev libgoogle-glog-dev libgflags-dev libspdlog-dev
RUN apt-get -y install libboost-dev libboost-serialization-dev libgl1-mesa-glx zlib1g-dev

RUN apt-get -y install libssl-dev libzip-dev

RUN ln -sf /usr/bin/python3.8 /usr/bin/python3

RUN python3 -m pip install --upgrade pip
RUN pip3 install 'cmake>=3.5' pybind11 pybind11-global numpy wheel opencv-python onnx tensorflow

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 50; \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 50; \
    update-alternatives --config gcc; \
    update-alternatives --config g++

RUN cd /usr/src/gtest; \
    cmake CMakeLists.txt; \
    make -j `nproc`; \
    cp *.a /usr/lib

# for demo
RUN pip3 install pillow matplotlib

RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v$PB_VERSION/protobuf-all-$PB_VERSION.tar.gz -P /tmp
RUN cd /tmp; \
    tar xvzf protobuf-all-$PB_VERSION.tar.gz
RUN cd /tmp/protobuf-$PB_VERSION; \
    ./configure; \
    make -j `nproc`; \
    make install; \
    ldconfig

RUN rm /tmp/protobuf-all-$PB_VERSION.tar.gz \
    rm -rf /tmp/protobuf-$PB_VERSION
