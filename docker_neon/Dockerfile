FROM ironbcc/demo1
MAINTAINER Stanislav Veretennikov <ironbcc@gmail.com>

######
# Neon
######

WORKDIR /root
RUN git clone https://github.com/NervanaSystems/neon.git && \
    cd neon && \
    make -j"$(nproc)" sysinstall

WORKDIR /root
RUN git clone https://github.com/NervanaSystems/caffe2neon.git && \
    pip install -r caffe2neon/requirements.txt

ENV PYTHONPATH $PYTHONPATH:$CAFFE_ROOT/python$CAFFE_ROOT/python/caffe/proto/
