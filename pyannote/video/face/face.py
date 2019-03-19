#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2015-2016 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Herve BREDIN - http://herve.niderb.fr
# Benjamin MAURICE - maurice@limsi.fr

"""Face processing"""

from __future__ import absolute_import
from __future__  import division
from __future__ import print_function
import cv2
import sys
import numpy as np
import mxnet as mx
import os
import sklearn
from sklearn.preprocessing import normalize
from pyannote.video.face.mtcnn_detector import MtcnnDetector
from skimage import transform as trans
from mxnet.contrib.onnx.onnx2mx.import_model import import_model

#print(os.path.dirname(os.path.abspath(__file__)))
#print(os.getcwd())

DLIB_SMALLEST_FACE = 36

def get_model(ctx, model):
    image_size = (112,112)
    # Import ONNX model
    sym, arg_params, aux_params = import_model(model)
    # Define and binds parameters to the network
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model

def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    # Assert input shape
    if len(str_image_size)>0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size)==1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size)==2
        assert image_size[0]==112
        assert image_size[0]==112 or image_size[1]==96

    # Do alignment using landmark points
    if landmark is not None:
        assert len(image_size)==2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32 )
        if image_size[1]==112:
            src[:,0] += 8.0
        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        assert len(image_size)==2
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
        return warped

    # If no landmark points available, do alignment using bounding box. If no bounding box available use center crop
    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret

def get_feature(model,aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()
    #embedding = sklearn.preprocessing.normalize(embedding).flatten()
    embedding = normalize(embedding).flatten()
    return embedding


class Face(object):
    """Face processing"""

    def __init__(self, landmarks=None, embedding=None):
        """Face detection

        Parameters
        ----------
        landmarks : str
            Path to MTCNN facial landmarks predictor model.
        embedding : str
            Path to ArcFace face embedding model.
        """
        super(Face, self).__init__()

        # Determine and set context
        if len(mx.test_utils.list_gpus())==0:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(0)

        # face detection
        if landmarks is not None:
            det_threshold = [0.6,0.7,0.8]
            self.face_detector_ = MtcnnDetector(model_folder=landmarks, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=det_threshold)

        # face embedding
        if embedding is not None:
            self.face_recognition_ = get_model(ctx , embedding)

    def get_debug(self, image, face, landmarks):
        """Return face with overlaid landmarks"""
        copy = image.copy()
        for p in landmarks:
            x, y = p[0], p[1]
            cv2.rectangle(copy, (x, y), (x, y), (0, 255, 0), 2)
        copy = copy[face.top():face.bottom(),
                    face.left():face.right()]
        copy = cv2.resize(copy, (self.size, self.size))
        return copy

    def iter_data(self, rgb, return_landmarks=False, return_embedding=False,
                    return_debug=False):

        ret = self.face_detector_.detect_face(rgb, det_type = 0)
        #print(ret is None, ret)
        if ret is not None:
            bbox, points = ret
            #print(bbox.shape)
            if bbox.shape[0]!=0:
                for id_f, face in enumerate(bbox):
                    # yield face if nothing else is asked for
                    if not (return_landmarks or return_embedding or return_debug):
                        yield face
                        continue

                    # always return face as first tuple element
                    result = (face, )

                    # compute landmarks
                    landmarks = []
                    mid_size = int(len(points[id_f])/2)
                    for i in range(mid_size):
                        landmarks.append((points[id_f][i], points[id_f][i+mid_size]))
                    landmarks = np.asarray(landmarks)
                    #[[135 166 147 127 158  95 101 108 128 135]] -> [(135,95), (166, 101), (147,108), (127,128), (158,135)]

                    # append landmarks
                    if return_landmarks:
                        result = result + (landmarks, )

                    # compute and append embedding
                    if return_embedding:
                        points_ = points[id_f,:].reshape((2,5)).T
                        # Call preprocess() to generate aligned images
                        nimg = preprocess(rgb, face, points_, image_size='112,112')
                        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                        aligned = np.transpose(nimg, (2,0,1))
                        embedding = get_feature(self.face_recognition_, aligned)
                        result = result + (embedding, )
                        #[[107.74201095  61.86504232 182.06073904 163.76174854   0.99981207]]

                    # compute and append debug image
                    if return_debug:
                        debug = self.get_debug(rgb, face, landmarks)
                        result = result + (debug, )

                    yield result
        return None

    def iterfaces(self, rgb):
        return self.iter_data(rgb)

    def __call__(self, rgb, return_landmarks=False, return_embedding=False,
                    return_debug=False):
        """Iterate over all faces

        Parameters
        ----------
        rgb : np.array
            RGB image to be processed
        return_landmarks : bool
            Whether to yield landmarks. Defaults to False.
        return_embedding : bool
            Whether to yield embedding. Defaults to False.
        return_debug : bool
            Whether to yield debugging image. Defaults to False.
        """

        return self.iter_data(rgb, return_landmarks, return_embedding, return_debug)
