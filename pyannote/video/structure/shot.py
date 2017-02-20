#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2015-2017 CNRS

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

import cv2
import itertools
import numpy as np
from ..video import Video
import scipy.signal
from pyannote.core import Segment

OPENCV = int(cv2.__version__.split('.')[0])


class Shot(object):
    """Shot boundary detection based on displaced frame difference

    Parameters
    ----------
    video : Video
    height : int, optional
        Resize video to this height, in pixels. Defaults to 50.
    context : float, optional
        Median filtering context (in seconds). Defaults to 2.
    threshold : float, optional
        Defaults to 1.
    """

    def __init__(self, video, height=50, context=2.0, threshold=1.0):
        super(Shot, self).__init__()
        self.video = video
        self.height = height
        self.threshold = threshold
        self.context = context

        # estimate new size from video size and target height
        w, h = self.video._size
        self._resize = (self.height, w * self.height / h)

        # estimate kernel size from context and video step
        kernel_size = self.context / self.video.step
        # kernel size must be an odd number greater than 3
        self._kernel_size = max(3, int(np.ceil(kernel_size) // 2 * 2 + 1))

        self._reconstruct = None

    def _convert(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, self._resize)

    def dfd(self, previous, current, flow=None):
        """Displaced frame difference"""

        if OPENCV == 2:
            flow = cv2.calcOpticalFlowFarneback(
                previous, current, 0.5, 3, 15, 3, 5, 1.1, 0, flow)
        elif OPENCV == 3:
            flow = cv2.calcOpticalFlowFarneback(
                previous, current, flow, 0.5, 3, 15, 3, 5, 1.1, 0)

        height, width = previous.shape

        # allocate "reconstruct" only once
        if self._reconstruct is None:
            self._reconstruct = np.empty(previous.shape)

        for x, y in itertools.product(range(width), range(height)):
            dy, dx = flow[y, x]
            rx = int(max(0, min(x + dx, width - 1)))
            ry = int(max(0, min(y + dy, height - 1)))
            self._reconstruct[y, x] = current[ry, rx]

        return np.mean(np.abs(previous - self._reconstruct))

    def iter_dfd(self):
        """Pairwise displaced frame difference"""

        previous = None

        # iterate frames one by one
        for t, rgb in self.video:

            current = self._convert(rgb)

            if previous is None:
                previous = current
                continue

            yield t, self.dfd(previous, current, flow=None)

            previous = current

    def __iter__(self):

        # TODO: running median
        t, y = zip(*self.iter_dfd())

        filtered = scipy.signal.medfilt(y, kernel_size=self._kernel_size)

        # normalized displaced frame difference
        normalized = (y - filtered) / filtered

        # apply threshold on normalized displaced frame difference
        # in case multiple consecutive value are higher than the threshold,
        # only keep the first one as a shot boundary.
        previous = self.video.start
        _i = 0
        for i in np.where(normalized > self.threshold)[0]:

            if i == _i + 1:
                _i = i
                continue

            yield Segment(previous, t[i])

            previous = t[i]
            _i = i

        yield Segment(previous, self.video.end)
