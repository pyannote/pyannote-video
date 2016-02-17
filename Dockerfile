FROM pyannote/core
MAINTAINER Hervé Bredin <bredin@limsi.fr>

ADD . /src
RUN pip install -e /src

# moviepy likes to download its own ffmpeg version at import time
# this command will guarantee this is done once and for all
RUN python -c "from moviepy.editor import VideoClip"

VOLUME /src

# update OpenCV because of https://github.com/Itseez/opencv/pull/6009
RUN mkdir /tmp/opencv_sources \
 && cd /tmp/opencv_sources \
 && wget https://github.com/Itseez/opencv/archive/master.zip \
 && unzip master.zip \
 && rm master.zip \
 && cd /tmp/opencv_sources \
 && wget https://github.com/Itseez/opencv_contrib/archive/master.zip \
 && unzip master.zip \
 && rm master.zip \
 && mkdir /tmp/opencv_sources/opencv-master/build \
 && cd /tmp/opencv_sources/opencv-master/build \
 && cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D INSTALL_C_EXAMPLES=OFF \
          -D INSTALL_PYTHON_EXAMPLES=ON \
          -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_sources/opencv_contrib-master/modules \
          -D BUILD_EXAMPLES=ON .. \
  && make -j2 \
  && make install \
  && ldconfig \
  && rm -rf /tmp/opencv_sources
