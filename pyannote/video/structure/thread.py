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
# Hervé BREDIN - http://herve.niderb.fr


from collections import deque
from itertools import combinations
import cv2
import networkx as nx
from pyannote.core import Annotation
from pyannote.core.time import _t_iter as getLabelGenerator
from tqdm import tqdm
import warnings

try:
    # Python 3
    from functools import lru_cache
except Exception as e:
    # Python 2
    from ..utils.lru_cache import lru_cache

def product_lookahead(iterable, lookahead):

    cache = deque([], lookahead + 1)
    for item in iterable:
        cache.append(item)

        if len(cache) < lookahead + 1:
            continue

        for j in range(lookahead):
            yield cache[0], cache[j+1]

    cache.popleft()
    for item1, item2 in combinations(cache, 2):
        yield item1, item2


class Thread(object):
    """Shot threading based on ORB features

    Can also be used as post-processing to clean up a (possibly over-segmented)
    segmentation into shots.

    """
    def __init__(self, video, shot=None, min_match=20, lookahead=24,
                 verbose=False):
        """
        Parameters
        ----------
        shot : iterable, optional
            Segment iterator.

        """
        super(Thread, self).__init__()

        self.video = video

        self.lookahead = lookahead
        if shot is None:
            shot = Shot(video)
        self.shot = shot

        self.verbose = verbose

        # ORB (non-patented SIFT alternative) extraction
        self._orb = cv2.ORB()

        # # brute-force ORB matching
        # self._bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # fast approximate nearest neighbord
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self._flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.min_match = min_match


    # _threads_graph method calls _compute_orb with the same "t" over and over.
    # we cache "maxsize" last calls to avoid recomputing ORB features
    @lru_cache(maxsize=128, typed=False)
    def _compute_orb(self, t):
        try:
            rgb = self.video(t)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            _, descriptors = self._orb.detectAndCompute(gray, None)

        except IOError as e:
            warnings.warn("unable to reach t = {t:.3f}".format(t=t))
            descriptors = None

        return descriptors

    def _match(self, orb1, orb2):
        """Check whether there is a match between orb1 and orb2"""

        if orb1 is None or orb2 is None:
            return False

        # matches = self._bfmatcher.match(orb1, orb2)
        matches = self._flann.knnMatch(orb1, orb2, k=2)

        count = 0

        for twoNearestNeighbors in matches:
            if len(twoNearestNeighbors) < 2:
                continue
            best, secondBest = twoNearestNeighbors
            if best.distance < 0.7 * secondBest.distance:
                count = count + 1

        return count > self.min_match

    def _threads_graph(self):

        # 5-frames collar
        collar = 5. / self.video._fps

        # build threading graph by comparing each shot
        # to 'lookahead' following shots
        threads = nx.Graph()

        generator = product_lookahead(self.shot, self.lookahead)
        if self.verbose:
            generator = tqdm(iterable=generator,
                             total=len(self.shot) * self.lookahead,
                             leave=True, mininterval=1.,
                             unit='shot pairs', unit_scale=True)

        for current, following in generator:
            orbLast = self._compute_orb(current.end - collar)
            orbFirst = self._compute_orb(following.start + collar)
            threads.add_node(current)
            if self._match(orbLast, orbFirst):
                threads.add_edge(current, following)
        threads.add_node(following)
        return threads

    def __call__(self):

        # list of chronologically sorted list of shots
        graph = self._threads_graph()
        threads = [sorted(cc) for cc in nx.connected_components(graph)]

        annotation = Annotation()
        labelGenerator = getLabelGenerator()

        # chronologically sorted threads (based on their first shot)
        for thread in sorted(threads, key=lambda thread: thread[0]):
            label = next(labelGenerator)
            for shot in thread:
                annotation[shot] = label

        return annotation.smooth()
