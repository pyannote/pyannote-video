FROM bamos/ubuntu-opencv-dlib-torch:ubuntu_14.04-opencv_2.4.11-dlib_18.16-torch_2016.05.07
MAINTAINER Hervé Bredin <bredin@limsi.fr>

# python package management
RUN DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    python \
    python-dev \
    python-pip

# scientific python
RUN pip install numpy
RUN pip install scipy
RUN pip install jupyter
RUN pip install matplotlib

# pyannote.core notebook support
RUN DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    graphviz \
    libgraphviz-dev
RUN pip install pyannote.core[notebook]

# pyannote.video ffmpeg
RUN add-apt-repository ppa:mc3man/trusty-media
RUN apt-get update
RUN apt-get install -yq ffmpeg

RUN pip install pyannote.video

# moviepy likes to download its own ffmpeg version at import time
# this command will guarantee this is done once and for all
RUN python -c "from moviepy.editor import VideoClip"

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0
RUN curl -L https://github.com/krallin/tini/releases/download/v0.6.0/tini > tini && \
    echo "d5ed732199c36a1189320e6c4859f0169e950692f451c03e7854243b95f4234b *tini" | sha256sum -c - && \
    mv tini /usr/bin/tini && \
    chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]


# run jupyter notebook by default
EXPOSE 8888
VOLUME /notebook
WORKDIR /notebook
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
