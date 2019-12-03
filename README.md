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

### Computed data format

We use [NumPy structured arrays](https://docs.scipy.org/doc/numpy/user/basics.rec.html) to store :
- bounding boxes formated as (left, top, right, bottom)
- identifier for the track
- time in seconds where that face was tracked
- status of the tracking algorithm, e.g. 'backward'
- landmarks
- embeddings

File which should be named as `<file_uri>.npy`.

The array has shape `(N,)`, with `N` being sum of the number of frames over every track.
Each track has dtype :
```py
[
  ('time', 'float64'),
  ('track', 'int64'),
  ('bbox', 'float64', (4,)),
  ('status', '<U21'),
  ('landmarks', 'float64', (68,)),
  ('embeddings', 'float64', (128,))
]
```

When using single rgb image (implemented in pyannote.db.plumcot), there's no tracking so each line of the array represents a different face (thus there's no 'time' nor 'track' nor 'status' field).
