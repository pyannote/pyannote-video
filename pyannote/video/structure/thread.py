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


from collections import deque
from itertools import combinations
import cv2
import networkx as nx
from pyannote.core import Annotation
from pyannote.core.time import _t_iter as getLabelGenerator
from tqdm import tqdm
import warnings
from pyannote.core.util import pairwise

OPENCV = int(cv2.__version__.split('.')[0])

try:
    # Python 3
    from functools import lru_cache
except Exception as e:
    # Python 2
    from ..utils.lru_cache import lru_cache


def product_lookahead(iterable, lookahead):

    cache = deque([], lookahead + 1)

    for item in iterable:

        # fill cache with up to 'lookahead + 1' items
        cache.append(item)
        if len(cache) < lookahead + 1:
            continue

        # iterate over lookahead pairs (*)
        for j in range(lookahead):
            yield cache[0], cache[j+1]

    # if cache is full, it means that the (*) part of the above loop
    # was executed and we should therefore not yield (cache[0], ...) pairs
    # a second time -- and thus remove cache[0] first.

    # reciprocally, if cache is not full, it means that the (*) part of the
    # above loop was never executed and we should therefore yield all possible
    # pairs, including (cach[0], ...) pairs -- and thus we do **not** remove
    # cache[0]

    if len(cache) == lookahead + 1:
        cache.popleft()

    # exhaust the cache with standard itertools combinations
    for item1, item2 in combinations(cache, 2):
        yield item1, item2


class Thread(object):
    """Shot threading based on ORB features

    Can also be used as post-processing to clean up a (possibly over-segmented)
    segmentation into shots.

    """
    def __init__(self, video, shot=None, height=200, min_match=20, lookahead=5,
                 verbose=False):
        """
        Parameters
        ----------
        shot : iterable, optional
            Segment iterator.

        """
        super(Thread, self).__init__()

        self.video = video
        self.height = height

        # estimate new size from video size and target height
        w, h = self.video._size
        self._resize = (self.height, w * self.height / h)

        self.lookahead = lookahead
        if shot is None:
            shot = Shot(video)
        self.shot = shot

        self.verbose = verbose

        # ORB (non-patented SIFT alternative) extraction
        if OPENCV == 2:
            self._orb = cv2.ORB()
        elif OPENCV == 3:
            self._orb = cv2.ORB_create()

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
            rgb = cv2.resize(self.video(t), self._resize)
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

        return count

    def _threads_graph(self):
        """Build and return thread graph

        Contains one node per shot.
        Shots `n` and `n+k` are connected iff the last frames
        of shot `n` are similar to the first frames of shot `n+k`
        (with k < lookahead)

        """

        shot = list(self.shot)

        # 10-frames collar
        collar = 10. / self.video.frame_rate

        # build threading graph by comparing each shot
        # to 'lookahead' following shots
        threads = nx.Graph()
        threads.add_nodes_from(shot)

        generator = product_lookahead(shot, self.lookahead)
        if self.verbose:
            generator = tqdm(iterable=generator,
                             total=len(shot) * self.lookahead,
                             leave=True, mininterval=1.,
                             unit='shot pairs', unit_scale=True)

        for current, following in generator:
            orbLast = self._compute_orb(current.end - collar)
            orbFirst = self._compute_orb(following.start + collar)
            n_matches = self._match(orbLast, orbFirst)
            if n_matches > self.min_match:
                threads.add_edge(current, following, n_matches=n_matches)

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

    def scenes(self, threads):

        g = nx.Graph()

        # connect adjacent shots
        for shot1, shot2 in pairwise(threads.itertracks()):
            g.add_edge(shot1, shot2)

        # connect threaded shots
        for label in threads.labels():
            for shot1, shot2 in pairwise(threads.subset([label]).itertracks()):
                g.add_edge(shot1, shot2)

        scenes = threads.copy()

        # group all shots of intertwined threads
        for shots in sorted(sorted(bc) for bc in nx.biconnected_components(g)):

            if len(shots) < 3:
                continue

            common_label = scenes[shots[0]]
            for shot in shots:
                scenes[shot] = common_label

        return scenes
