### Version 1.6 (2018-05-22)

  - improve: simplify installation (latest dlib, pip's opencv-python, & ffmpeg from imageio)
  - fix: switch to pyannote.algorithms 0.8
  - fix: fix shot theading
  - doc: update "Getting started" guide

### Version 1.5.1 (2017-08-09)

  - fix: use latest pandas API (@crhan & @epratheeban)
  - fix: use latest moviepy (@crhan)
  - doc: update README
  - setup: use latest pyannote.core

### Version 1.5 (2017-03-29)

  - feat: switch from Openface to dlib
  - fix: fix Python 3 support
  - doc: update 'Getting started' notebook

### Version 1.4 (2017-02-20)

  - fix: fix face clustering
  - fix: fix support for latest numpy
  - fix: fix Docker build
  - setup: switch to pyannote.core 0.13
  - setup: switch to pyannote.algorithms 0.7.3
  - setup: switch to dlib 19.1.0

### Version 1.3.13 (2016-11-08)

  - fix: fix Docker build using latest pip version
  - setup: switch to pyannote.algorithms 0.6.6

### Version 1.3.12 (2016-11-05)

  - fix: fix Docker build (@yaochx)
  - fix: fix getting started guide (@yaochx)

### Version 1.3.11 (2016-06-27)

  - setup: switch to pyannote.core 0.6.6
  - feat: add --torch parameter to pyannote-face.py script
  - feat: pseudo-scenes segmentation based on shot threads
  - doc: "getting started" notebook

### Version 1.3.9 (2016-06-13)

  - fix: include openface_server.lua in package

### Version 1.3.8 (2016-05-12)

  - fix: corner case in one-shot threading

### Version 1.3.6 (2016-05-11)

  - fix: remove hard-coded path to torch
  - fix: corner case with iterable shorter than lookahead
  - setup: update pyannote.algorithms dependency
  - fix: include openface_server.lua in package

### Version 1.3.1 (2016-03-25)

  - fix: pyannote.core breaking changes

### Version 1.3 (2016-02-17)

  - feat: face clustering
  - fix: OpenCV 2/3 support
  - improve: shot threading
  - improve: faster face tracking-by-detection

### Version 1.2.1 (2016-01-20)

  - fix: __iter__ would incorrectly raise an IOError at the end of some Videos
  - docker: pre-fetch MoviePy's own ffmpeg

### Version 1.2 (2016-01-15)

  - feat: face processing
  - feat: shot threading

### Version 1.1 (2015-10-12)

  - feat: shot boundary detection (pyannote-shot)

### Version 1.0 (2015-10-09)

  - first public release
