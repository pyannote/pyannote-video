# Announcement

Open [postdoc position](https://mycore.core-cloud.net/public.php?service=files&t=2b5f5a79d24ac81c3b3c371fcd80734b) at [LIMSI](https://www.limsi.fr/en/) combining machine learning, NLP, speech processing, and computer vision.

# pyannote-video

> a toolkit for face detection, tracking, and clustering in videos

## Installation

See https://github.com/onnx/models/blob/master/models/face_recognition/ArcFace/arcface_inference.ipynb to install ArcFace and MTCNN models

In python :
```python
import sys
import mxnet as mx
import os
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from mxnet.contrib.onnx.onnx2mx.import_model import import_model

PATH_DATA=''

# Download onnx model
mx.test_utils.download(dirname=PATH_DATA, url='https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100.onnx')

for i in range(4):
    mx.test_utils.download(dirname=PATH_DATA+'mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}-0001.params'.format(i+1))
    mx.test_utils.download(dirname=PATH_DATA+'mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}-symbol.json'.format(i+1))
    mx.test_utils.download(dirname=PATH_DATA+'mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}.caffemodel'.format(i+1))
    mx.test_utils.download(dirname=PATH_DATA+'mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}.prototxt'.format(i+1))
```

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

When you launch `pyannote_face.py extract`, ther arguments are the movie, the file .track.txt, the path to the folder mtcnn-model, the model resnet100.onnx, the file .landmarks.txt and the file .embedding.txt
