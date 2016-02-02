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
# Herve BREDIN - http://herve.niderb.fr

"""Face tracking"""

from .face import Face
from .face import DLIB_SMALLEST_FACE
from ..tracking import TrackingByDetection


def get_face_detect(face):
    """Create function for face detection"""
    def face_detect(frame):
        """Detect face in frame"""
        for f in face.iterfaces(frame):
            yield (f.left(), f.top(), f.right(), f.bottom())
    return face_detect


class FaceTracking(TrackingByDetection):
    """Face tracking

    Parameters
    ----------
    detect_min_size : float, optional
        Approximate size (in video height ratio) of the smallest face that
        should be detected. Defaults to any face.
    detect_every : float, optional
        When provided, face detection is applied every `detect_every` seconds.
        Defaults to processing every frame.
    track_min_confidence : float, optional
        Kill trackers whose confidence goes below this value. Defaults to 10.
    track_min_overlap_ratio : float, optional
        Do not associate trackers and detections if their overlap ratio goes
        below this value. Defaults to 0.3.
    track_max_gap : float, optional
        Bridge gaps with duration shorter than this value.
    """
    def __init__(self, detect_min_size=0., detect_every=0.,
                 track_min_confidence=10.,track_min_overlap_ratio=0.3,
                 track_max_gap=0.):

        face = Face()
        detect_func = get_face_detect(face)

        super(FaceTracking, self).__init__(
            detect_func=detect_func,
            detect_smallest=DLIB_SMALLEST_FACE,
            detect_min_size=detect_min_size,
            detect_every=detect_every,
            track_min_confidence=track_min_confidence,
            track_min_overlap_ratio=track_min_overlap_ratio,
            track_max_gap=track_max_gap)
