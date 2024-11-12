#
# base of build-container
#
FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04 
MAINTAINER jieun.lim <jieun.lim@sapeon.com>

# set non-interative mode while installing apts
ARG DEBIAN_FRONTEND=noninteractive

# TimeZone 
ENV TZ Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install packages
RUN apt-get update && apt-get install -y \
  build-essential pkg-config software-properties-common \
  lsb-release \
  sudo unzip wget curl \
  git git-core git-flow \
  gcc g++ make cmake meson ninja-build \
  vim tmux zsh \
  python3-pip \
  libgoogle-glog-dev \
  libnsync-dev \
  && rm -rf /var/lib/apt/lists/*;

# install python modules
RUN pip3 install opencv-python \
  google protobuf==3.20.1 \
  cpplint \
  torch torchvision torchaudio torchmetrics \ 
  --extra-index-url https://download.pytorch.org/whl/cu113 

#################################################
# for aixgraph_simulator

WORKDIR $HOME

# install protobuf
RUN sudo apt-get update \
  && sudo apt-get install -y autoconf automake libtool curl make g++ unzip \
  && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.20.1/protobuf-cpp-3.20.1.tar.gz \
  && tar -zxvf protobuf-cpp-3.20.1.tar.gz \
  && cd protobuf-3.20.1 \
  && ./configure --prefix=/usr \
  && make -j 16 \
  #&& make check \ 
  && sudo make install \
  && sudo ldconfig \
  && cd ..

# install boost 1.71.0
RUN sudo apt-get install -y libboost-all-dev

# install OpenCV C++
# RUN mkdir -p opencv \
#   && cd opencv \
#   && sudo apt update && sudo apt install -y cmake g++ wget unzip \
#   && wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip \
#   && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip \
#   && unzip opencv.zip \
#   && unzip opencv_contrib.zip \
#   && mkdir -p build && cd build \ 
#   && cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x \
#   && cmake --build . -j 16 \
#   && sudo make install \ 
#   && cd ../..

# install googletest
RUN git clone https://github.com/google/googletest.git -b v1.12.x \
  && cd googletest \
  && mkdir build && cd build \
  && cmake -DBUILD_SHARED_LIBS=ON .. \
  && make -j 16 \
  && sudo make install \
  && sudo ldconfig \
  && cd ../..
