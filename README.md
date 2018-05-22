# Announcement

Open [postdoc position](https://mycore.core-cloud.net/public.php?service=files&t=2b5f5a79d24ac81c3b3c371fcd80734b) at [LIMSI](https://www.limsi.fr/en/) combining machine learning, NLP, speech processing, and computer vision.

# pyannote-video

> a toolkit for face detection, tracking, and clustering in videos

## Installation

Create a new `conda` environment:

```bash
$ conda create -n pyannote python=3.6 anaconda
$ source activate pyannote
```

Install `pyannote-video` and its dependencies:

```bash
$ pip install pyannote-video
```

Download `dlib` models:

```bash
$ git clone https://github.com/pyannote/pyannote-data.git
$ git clone https://github.com/davisking/dlib-models.git
$ bunzip2 dlib-models/dlib_face_recognition_resnet_model_v1.dat.bz2
$ bunzip2 dlib-models/shape_predictor_68_face_landmarks.dat.bz2
```

## Tutorial

To execute the ["Getting started"](http://nbviewer.ipython.org/github/pyannote/pyannote-video/blob/master/doc/getting_started.ipynb) notebook locally, download the example video and `pyannote.video` source code:

```bash
$ git clone https://github.com/pyannote/pyannote-data.git
$ git clone https://github.com/pyannote/pyannote-video.git
$ pip install jupyter
$ jupyter notebook --notebook-dir="pyannote-video/doc"
```

## Documentation

No proper documentation for the time being...
