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
from .openface import TorchWrap

DLIB_SMALLEST_FACE = 36

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

OUTER_EYES_AND_NOSE = np.array([36, 45, 33])

class Face(object):
    """Face processing"""
    def __init__(self, landmarks=None, openface=None, size=96, torch='th'):
        """Face detection

        Parameters
        ----------
        landmarks : str
            Path to dlib's 68 facial landmarks predictor model.
        size : int
            Size of the normalized face thumbnail.
        openface : str
            Path to openface FaceNet model.
        """
        super(Face, self).__init__()

        # face detection
        self._face_detector = dlib.get_frontal_face_detector()

        # landmark detection
        if landmarks is not None:
            self._landmarks_detector = dlib.shape_predictor(landmarks)

        # normalization
        self.size = size
        self._landmarks = MINMAX_TEMPLATE * self.size

        if openface is not None:
            self._net = TorchWrap(torch=torch, model=openface, size=self.size, cuda=False)

    # face detection

    def iterfaces(self, rgb):
        """Iterate over all detected faces"""
        for face in self._face_detector(rgb, 1):
            yield face

    # landmarks detection

    def _get_landmarks(self, rgb, face):
        """Return facial landmarks"""
        points = self._landmarks_detector(rgb, face).parts()
        landmarks = np.float32([(p.x, p.y) for p in points])
        return landmarks

    def landmarks(self, rgb, face):
        """Return facial landmarks"""
        return self._get_landmarks(rgb, face)

    # face normalization

    def _get_normalized(self, rgb, landmarks):
        """Return normalized face based on outer eyes and nose tip"""
        matrix = cv2.getAffineTransform(
            landmarks[OUTER_EYES_AND_NOSE],
            self._landmarks[OUTER_EYES_AND_NOSE])
        normalized = cv2.warpAffine(rgb, matrix, (self.size, self.size))
        return normalized

    def normalize(self, rgb, face):
        """Return normalized face"""
        landmarks = self._get_landmarks(rgb, face)
        return self._get_normalized(rgb, landmarks)

    # openface feature extraction

    def _get_openface(self, bgr):
        """(internal) openface feature extraction"""
        return self._net.forwardImage(bgr)

    def openface(self, rgb, face):
        """Return Openface features"""
        normalized_rgb = self.normalize(rgb, face)
        normalized_bgr = cv2.cvtColor(normalized_rgb, cv2.COLOR_BGR2RGB)
        return self._get_openface(normalized_bgr)

    # debugging

    def debug(self, image, face, landmarks):
        """Return face with overlaid landmarks"""
        copy = image.copy()
        for x, y in landmarks:
            cv2.rectangle(copy, (x, y), (x, y), (0, 255, 0), 2)
        copy = copy[face.top():face.bottom(),
                    face.left():face.right()]
        copy = cv2.resize(copy, (self.size, self.size))
        return copy

    def __call__(self, rgb, return_landmarks=False, return_normalized=False,
                 return_openface=False, return_debug=False):
        """Iterate over all faces

        Parameters
        ----------
        rgb : np.array
            RGB image to be processed
        return_landmarks : bool
            Whether to yield landmarks. Defaults to False.
        return_normalized : bool
            Whether to yield normalized face. Defaults to False.
        return_openface : bool
            Whether to yield openface descriptor. Defaults to False.
        return_debug : bool
            Whether to yield debugging image. Defaults to False.
        """

        for face in self.iterfaces(rgb):

            _face = (face.left(), face.top(), face.right(), face.bottom())

            # yield face if nothing else is asked for
            if not (return_landmarks or return_normalized or
                    return_openface or return_debug):
                yield _face
                continue

            # always return face as first tuple element
            result = (_face, )

            # compute landmarks
            landmarks = self._get_landmarks(rgb, face)

            # normalize face
            if return_normalized or return_openface:
                normalized = self._get_normalized(rgb, landmarks)

            # compute openface descriptor
            if return_openface:
                openface = self._get_openface(normalized)

            # compute debugging image
            if return_debug:
                debug = self.debug(rgb, face, landmarks)

            # append landmarks
            if return_landmarks:
                result = result + (landmarks, )

            # append normalized face
            if return_normalized:
                result = result + (normalized, )

            # append openface descriptor
            if return_openface:
                result = result + (openface, )

            # append debugging image
            if return_debug:
                result = result + (debug, )

            yield result
