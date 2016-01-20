FROM pyannote/base
MAINTAINER Hervé Bredin <bredin@limsi.fr>

ADD . /src
RUN pip install -e /src

# moviepy likes to download its own ffmpeg version at import time
# this command will guarantee this is done once and for all
RUN python -c "from moviepy.editor import VideoClip"

VOLUME /src
