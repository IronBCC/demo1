FROM bamos/openface
MAINTAINER Stanislav Veretennikov <ironbcc@gmail.com>

RUN pip install \
        ipykernel \
        jupyter \
    	click \
        matplotlib \
	easydict \
        && \
    python -m ipykernel.kernelspec
########
# CUDA 8
########
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +2 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/cuda.list

ENV CUDA_VERSION 8.0
LABEL com.nvidia.cuda.version="8.0"

ENV CUDA_PKG_VERSION 8-0=8.0.44-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvrtc-$CUDA_PKG_VERSION \
        cuda-nvgraph-$CUDA_PKG_VERSION \
        cuda-cusolver-$CUDA_PKG_VERSION \
        cuda-cublas-$CUDA_PKG_VERSION \
        cuda-cufft-$CUDA_PKG_VERSION \
        cuda-curand-$CUDA_PKG_VERSION \
        cuda-cusparse-$CUDA_PKG_VERSION \
        cuda-npp-$CUDA_PKG_VERSION \
        cuda-cudart-$CUDA_PKG_VERSION \
	nvidia-367 && \
    ln -s cuda-$CUDA_VERSION /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

###CUDA-devel
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-core-$CUDA_PKG_VERSION \
        cuda-misc-headers-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-nvrtc-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-nvgraph-dev-$CUDA_PKG_VERSION \
        cuda-cusolver-dev-$CUDA_PKG_VERSION \
        cuda-cublas-dev-$CUDA_PKG_VERSION \
        cuda-cufft-dev-$CUDA_PKG_VERSION \
        cuda-curand-dev-$CUDA_PKG_VERSION \
        cuda-cusparse-dev-$CUDA_PKG_VERSION \
        cuda-npp-dev-$CUDA_PKG_VERSION \
        cuda-cudart-dev-$CUDA_PKG_VERSION \
        cuda-driver-dev-$CUDA_PKG_VERSION && \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs:${LIBRARY_PATH}

###CUDA-CuDNN
RUN apt-get update && apt-get install -y \
        curl && \
    rm -rf /var/lib/apt/lists/*

ENV CUDNN_VERSION 5
LABEL com.nvidia.cudnn.version="5"

RUN CUDNN_DOWNLOAD_SUM=a87cb2df2e5e7cc0a05e266734e679ee1a2fadad6f06af82a76ed81a23b102c8 && \
    curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz -O && \
    echo "$CUDNN_DOWNLOAD_SUM  cudnn-8.0-linux-x64-v5.1.tgz" | sha256sum -c --strict - && \
    tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local && \
    rm cudnn-8.0-linux-x64-v5.1.tgz && \
    ldconfig

#######
# CAFFE
#######
ENV CAFFE_ROOT=/root/caffe

RUN apt-get update && \
apt-get install  -y --no-install-recommends \
	libprotobuf-dev \
	libleveldb-dev \
	libsnappy-dev \
	libopencv-dev \
	libhdf5-serial-dev \
	protobuf-compiler \
	libgflags-dev \
	libgoogle-glog-dev \
	liblmdb-dev \
	libatlas-base-dev \
	libatlas-dev \
&& apt-get install --no-install-recommends libboost-all-dev

WORKDIR $CAFFE_ROOT
RUN git clone -b master --depth 1 https://github.com/BVLC/caffe.git . && \
    pip install --no-use-wheel scikit-image && \
    for req in $(cat python/requirements.txt) pydot; do pip install $req; done 

RUN cp Makefile.config.example Makefile.config && \
    echo CXXFLAGS += -std=c++11 >> Makefile.config && \   
    make -j"$(nproc)" all 

#RUN cp Makefile.config.example Makefile.config && \
#    echo USE_CUDNN := 0 >> Makefile.config && \ 
#    echo CXXFLAGS += -std=c++11 >> Makefile.config && \   
#    make -j"$(nproc)" all 

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR $CAFFE_ROOT
RUN make -j"$(nproc)" pycaffe

RUN apt-get update && apt-get install -y --no-install-recommends \
        libopencv-dev \
        python-opencv \
	python-matplotlib \
	python-numpy

###########
# MXNET-SSD
###########
WORKDIR /root
RUN git clone --recursive https://github.com/zhreshold/mxnet-ssd.git

ENV SSD_ROOT=/root/mxnet-ssd
ENV MXNET_ROOT=/root/mxnet-ssd/mxnet

WORKDIR $SSD_ROOT
RUN wget -q https://dl.dropboxusercontent.com/u/39265872/ssd_300_voc0712.zip && \
	unzip ssd_300_voc0712.zip -d model && \
	rm ssd_300_voc0712.zip

RUN apt-get update && apt-get install -y build-essential git libopenblas-dev libopencv-dev
RUN cd mxnet && \
    cp make/config.mk . && \
    echo "USE_CUDA=1" >>config.mk && \
    echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk && \
    echo "USE_CUDNN=1" >>config.mk && \
    echo "USE_BLAS=atlas" >>config.mk && \
    make -j$(nproc) ADD_LDFLAGS=-L/usr/local/cuda/lib64/stubs
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH 

#############
# DEEPCUT-CNN
#############
ENV DEEPCUT_ROOT=/root/deepcut-cnn

WORKDIR /root
RUN git clone https://github.com/eldar/deepcut-cnn && \
	cd $DEEPCUT_ROOT/models/deepercut && \
	./download_models.sh

#########
# Node.js
#########
#FROM node
RUN groupadd --gid 1000 node \
  && useradd --uid 1000 --gid node --shell /bin/bash --create-home node

# gpg keys listed at https://github.com/nodejs/node
RUN set -ex \
  && for key in \
    9554F04D7259F04124DE6B476D5A82AC7E37093B \
    94AE36675C464D64BAFA68DD7434390BDBE9B9C5 \
    0034A06D9D9B0064CE8ADF6BF1747F4AD2306D93 \
    FD3A5288F042B6850C66B31F09FE44734EB7990E \
    71DCFD284A79C3B38668286BC97EC7A07EDE3FC1 \
    DD8F2338BAE7501E3DD5AC78C273792F7D83545D \
    B9AE9905FFD7803F25714661B63B535A4C206CA9 \
    C4F0DFFF4E8C1A8236409D08E73BC641CC11F4C8 \
  ; do \
    gpg --keyserver ha.pool.sks-keyservers.net --recv-keys "$key"; \
  done

ENV NPM_CONFIG_LOGLEVEL info
ENV NODE_VERSION 7.4.0

RUN curl -SLO "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz" \
  && curl -SLO "https://nodejs.org/dist/v$NODE_VERSION/SHASUMS256.txt.asc" \
  && gpg --batch --decrypt --output SHASUMS256.txt SHASUMS256.txt.asc \
  && grep " node-v$NODE_VERSION-linux-x64.tar.xz\$" SHASUMS256.txt | sha256sum -c - \
  && tar -xJf "node-v$NODE_VERSION-linux-x64.tar.xz" -C /usr/local --strip-components=1 \
  && rm "node-v$NODE_VERSION-linux-x64.tar.xz" SHASUMS256.txt.asc SHASUMS256.txt \
  && ln -s /usr/local/bin/node /usr/local/bin/nodejs

WORKDIR /root
RUN apt-get update && apt-get install -y zip && \
    wget -q https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip && \
    unzip ngrok-stable-linux-amd64.zip && \
    mv ngrok /usr/bin

#######
# END
#######
EXPOSE 8000 9000 8888 8080
CMD /bin/bash
