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

"""Face processing"""

import numpy as np
import dlib
import cv2

DLIB_SMALLEST_FACE = 36


class Face(object):
    """Face processing"""

    def __init__(self, landmarks=None, embedding=None):
        """Face detection

        Parameters
        ----------
        landmarks : str
            Path to dlib's 68 facial landmarks predictor model.
        embedding : str
            Path to dlib's face embedding model.
        """
        super(Face, self).__init__()

        # face detection
        self.face_detector_ = dlib.get_frontal_face_detector()

        # landmark detection
        if landmarks is not None:
            self.shape_predictor_ = dlib.shape_predictor(landmarks)

        # face embedding
        if embedding is not None:
            self.face_recognition_ = dlib.face_recognition_model_v1(embedding)

    def iterfaces(self, rgb):
        """Iterate over all detected faces"""
        for face in self.face_detector_(rgb, 1):
            yield face

    def get_landmarks(self, rgb, face):
        return self.shape_predictor_(rgb, face)
        #return np.float32([(p.x, p.y) for p in landmarks.parts()])

    def get_embedding(self, rgb, landmarks):
        embedding = self.face_recognition_.compute_face_descriptor(
            rgb, landmarks)
        return embedding

    def get_debug(self, image, face, landmarks):
        """Return face with overlaid landmarks"""
        copy = image.copy()
        for p in landmarks.parts():
            x, y = p.x, p.y
            cv2.rectangle(copy, (x, y), (x, y), (0, 255, 0), 2)
        copy = copy[face.top():face.bottom(),
                    face.left():face.right()]
        copy = cv2.resize(copy, (self.size, self.size))
        return copy

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

        for face in self.iterfaces(rgb):

            # yield face if nothing else is asked for
            if not (return_landmarks or return_embedding or return_debug):
                yield face
                continue

            # always return face as first tuple element
            result = (face, )

            # compute landmarks
            landmarks = self.get_landmarks(rgb, face)

            # append landmarks
            if return_landmarks:
                result = result + (landmarks, )

            # compute and append embedding
            if return_embedding:
                embedding = self.get_embedding(rgb, landmarks)
                result = result + (embedding, )

            # compute and append debug image
            if return_debug:
                debug = self.get_debug(rgb, face, landmarks)
                result = result + (debug, )

            yield result
