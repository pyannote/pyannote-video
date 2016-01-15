FROM pyannote/base
MAINTAINER Hervé Bredin <bredin@limsi.fr>

ADD . /src
RUN pip install -e /src

VOLUME /src
