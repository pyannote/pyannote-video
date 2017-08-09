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
# from pyannote.algorithms.clustering.hac.constraint import DoNotCooccur

from scipy.spatial.distance import euclidean, pdist, cdist, squareform


class _Model(HACModel):
    """Average Euclidean distance between face embeddings"""

    def __init__(self):
        super(_Model, self).__init__(is_symmetric=True)

    @staticmethod
    def _to_segment(group):
        return Segment(np.min(group.time), np.max(group.time))

    def preprocess(self, embedding):
        """
        Parameters
        ----------
        embedding : str
            Path to face embeddings
        """

        # TODO : option to only keep 'detections'
        # (make sure it does not alter 'starting_point' segments)

        names = ['time', 'track']
        for i in range(128):
            names += ['d{0}'.format(i)]
        data = read_table(embedding, delim_whitespace=True,
                          header=None, names=names)
        data.sort_values(by=['track', 'time'], inplace=True)
        starting_point = Annotation(modality='face')
        for track, segment in data.groupby('track').apply(self._to_segment).iteritems():
            if not segment:
                continue
            starting_point[segment, track] = track

        return starting_point, data

    def compute_model(self, cluster, parent=None):
        # this method assumes that parent.features has been obtained
        # using the preprocess method
        return np.where(parent.features['track'] == cluster)[0]

    def compute_merged_model(self, clusters, parent=None):
        return np.hstack(self[cluster] for cluster in clusters)

    def compute_similarity_matrix(self, parent=None):

        # name and number of clusters
        clusters = list(self._models)
        n_clusters = len(clusters)

        # precompute pairwise embedding distance
        data = parent.features
        X = np.array(data[data.columns[2:]])
        self.precomputed_ = -squareform(pdist(X, metric='euclidean'))

        matrix = ValueSortedDict()
        for i, j in itertools.combinations(range(n_clusters), 2):
            # indices of embedding in ith cluster
            indices_i = self[clusters[i]]
            # indices of embedding in jth cluster
            indices_j = self[clusters[j]]
            # mean of all pairwise euclidean distances
            similarity = np.mean(self.precomputed_[indices_i][:, indices_j])
            matrix[clusters[i], clusters[j]] = similarity
            matrix[clusters[j], clusters[i]] = similarity

        return matrix

    def compute_similarity(self, cluster1, cluster2, parent=None):
        indices_1 = self[cluster1]
        indices_2 = self[cluster2]
        return np.mean(self.precomputed_[indices_1][:, indices_2])


class FaceClustering(HierarchicalAgglomerativeClustering):
    """Face clustering

    Parameters
    ----------
    threshold : float, optional
        Defaults to 0.6.

    Usage
    -----
    >>> clustering = FaceClustering()
    >>> starting_point, features = clustering.model.preprocess(embedding)
    >>> result = clustering(starting_point, features=features)

    """

    def __init__(self, threshold=0.6, force=False, logger=None):
        model = _Model()
        stopping_criterion = DistanceThreshold(threshold=threshold,
                                               force=force)
        # constraint = DoNotCooccur()
        constraint = None
        super(FaceClustering, self).__init__(
            model,
            stopping_criterion=stopping_criterion,
            constraint=constraint,
            logger=logger)
