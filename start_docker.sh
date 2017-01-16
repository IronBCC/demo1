#!/usr/bin/env bash
sudo nvidia-docker run -it --device /dev/video0 -v /home/ironbcc/cherry/demo1:/root/demo1 ironbcc/demo1

#  -e DISPLAY=$DISPLAY \
#  --env="QT_X11_NO_MITSHM=1" \
#  --privileged -v /dev/video0:/dev/video0 \
# --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro  \