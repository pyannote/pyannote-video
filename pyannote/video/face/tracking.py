#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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
# Hervé BREDIN - http://herve.niderb.fr

"""Face tracking by detection"""

from .face import Face
from .face import SMALLEST_DEFAULT
from ..tracking import TrackingByDetection


def get_face_detect(face):
    """Create function for face detection"""
    def face_detect(frame):
        """Detect face in frame"""
        for f in face.iterfaces(frame):
            yield (f.left(), f.top(), f.right(), f.bottom())
    return face_detect


class FaceTracking(TrackingByDetection):
    """
    """
    def __init__(self, smallest=SMALLEST_DEFAULT,
                 min_confidence=10., min_overlap_ratio=0.3, max_gap=0.):

        face = Face(smallest=smallest)
        detect_func = get_face_detect(face)

        super(FaceTracking, self).__init__(
            detect_func=detect_func,
            min_confidence=min_confidence,
            min_overlap_ratio=min_overlap_ratio,
            max_gap=max_gap)
