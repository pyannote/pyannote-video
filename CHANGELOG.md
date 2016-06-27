### Version 1.3.10 (2016-06-27)

  - setup: switch to pyannote.core 0.6.6
  - feat: add --torch parameter to pyannote-face.py script
  - feat: pseudo-scenes segmentation based on shot threads

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
