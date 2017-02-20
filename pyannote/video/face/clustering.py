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


from __future__ import print_function
from __future__ import unicode_literals

import itertools
import numpy as np
from sortedcollections import ValueSortedDict

from pandas import read_table
from pyannote.core import Segment, Annotation

from pyannote.algorithms.clustering.hac import \
    HierarchicalAgglomerativeClustering
from pyannote.algorithms.clustering.hac.model import HACModel
from pyannote.algorithms.clustering.hac.stop import DistanceThreshold
from pyannote.algorithms.clustering.hac.constraint import DoNotCooccur

from scipy.spatial.distance import euclidean, pdist, cdist, squareform

def l2_normalize(fX):

    norm = np.sqrt(np.sum(fX ** 2, axis=1))
    norm[norm == 0] = 1.
    return (fX.T / norm).T



class _Model(HACModel):
    """Euclidean distance between (weighted) average descriptor"""

    def __init__(self):
        super(_Model, self).__init__(is_symmetric=True)

    @staticmethod
    def _to_segment(group):
        return Segment(np.min(group.time), np.max(group.time))

    def preprocess(self, openface):
        """
        Parameters
        ----------
        openface : str
            Path to Openface features
        """

        # TODO : option to only keep 'detections'
        # (make sure it does not alter 'starting_point' segments)

        names = ['time', 'track']
        for i in range(128):
            names += ['d{0}'.format(i)]
        data = read_table(openface, delim_whitespace=True,
                          header=None, names=names)
        features = data.groupby('track')
        starting_point = Annotation(modality='face')
        for track, segment in features.apply(self._to_segment).iteritems():
            if not segment:
                continue
            starting_point[segment, track] = track

        return starting_point, features

    def compute_model(self, cluster, parent=None):

        # this method assumes that parent.features has been obtained
        # using the preprocess method

        # gather features from all tracks already clusters
        X = []
        for _, track in parent.current_state.subset([cluster]).itertracks():
            X.append(np.array(parent.features.get_group(track))[:, 2:])
        X = np.vstack(X)

        n = len(X)
        x = np.average(X, axis=0)

        return (x, n)

    def compute_merged_model(self, clusters, parent=None):

        X, N = zip(*[self[cluster] for cluster in clusters])

        x = np.average(X, axis=0, weights=N)
        n = np.sum(N)

        return (x, n)

    def compute_similarity_matrix(self, parent=None):

        clusters = list(self._models)
        n_clusters = len(clusters)

        X = np.vstack([self[cluster][0] for cluster in clusters])

        nX = l2_normalize(X)
        similarities = -squareform(pdist(X, metric='euclidean'))

        matrix = ValueSortedDict()
        for i, j in itertools.combinations(range(n_clusters), 2):
            matrix[clusters[i], clusters[j]] = similarities[i, j]
            matrix[clusters[j], clusters[i]] = similarities[j, i]

        return matrix

    def compute_similarities(self, cluster, clusters, parent=None):

        x = self[cluster][0].reshape((1, -1))
        X = np.vstack([self[c][0] for c in clusters])

        # L2 normalization
        nx = l2_normalize(x)
        nX = l2_normalize(X)

        similarities = -cdist(nx, nX, metric='euclidean')

        matrix = ValueSortedDict()
        for i, cluster_ in enumerate(clusters):
            matrix[cluster, cluster_] = similarities[0, i]
            matrix[cluster_, cluster] = similarities[0, i]

        return matrix

    def compute_similarity(self, cluster1, cluster2, parent=None):

        x1, _ = self[cluster1]
        x2, _ = self[cluster2]

        nx1 = l2_normalize(x1)
        nx2 = l2_normalize(x2)

        similarities = -cdist([nx1], [nx2], metric='euclidean')
        return similarities[0, 0]


class FaceClustering(HierarchicalAgglomerativeClustering):
    """Face clustering

    Parameters
    ----------
    threshold : float, optional
        Defaults to 1.0.

    Usage
    -----
    >>> clustering = FaceClustering()
    >>> # openface = path to Openface features
    >>> starting_point, features = clustering.model.preprocess(openface)
    >>> result = clustering(starting_point, features=features)

    """

    def __init__(self, threshold=1.0, force=False, logger=None):
        model = _Model()
        stopping_criterion = DistanceThreshold(threshold=threshold,
                                               force=force)
        constraint = DoNotCooccur()
        super(FaceClustering, self).__init__(
            model,
            stopping_criterion=stopping_criterion,
            constraint=constraint,
            logger=logger)
