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

"""Tracking by detection"""

from __future__ import division
import itertools
import numpy as np
import networkx as nx
from munkres import Munkres
import dlib

FORWARD = 'forward'
BACKWARD = 'backward'
DETECTION = 'detection'
ERROR = 'error'


def get_segment_generator(segmentation):
    """Time-driven segment generator"""
    t = yield
    for segment in segmentation:
        T = segment.end

        while True:
            # loop until a large enough t is sent to the generator
            if T > t:
                t = yield
                continue

            # else, we found a new segment
            t = yield T
            break


def get_min_max_t(track):
    """Get track stat and end times"""
    m = min(t for t, _, _ in track)
    M = max(t for t, _, _ in track)
    return (m, M)


class TrackingByDetection(object):
    """(Forward/backward) tracking by detection

    Parameters
    ----------
    detect_func : func
        Detection function. Should take video frame as input and return list
        (or iterable) of detections as (left, top, right, bottom) tuples.
    detect_smallest : int, optional
        Smallest object (height, in pixels) that `detect_func` can detect.
        Defaults to any object.
    detect_min_size : float, optional
        Approximate size (in video height ratio) of the smallest object that
        should be detected. Defaults to any object.
    detect_every : float, optional
        When provided, `detect_func` is applied every `detect_every` seconds.
        Defaults to processing every frame.
    track_min_confidence : float, optional
        Kill trackers whose confidence goes below this value. Defaults to 10.
    track_min_overlap_ratio : float, optional
        Do not associate trackers and detections if their overlap ratio goes
        below this value. Defaults to 0.3.
    track_max_gap : float, optional
        Bridge gaps with duration shorter than this value.

    Usage
    -----
    >>> from pyannote.video import Video, TrackingByDetection
    >>> video = Video(path_to_video)
    >>> # load segmentation into shots
    >>> tracking = TrackingByDetection()
    >>> for face_track in tracking(video, shots):
    ...     # do something with face track
    ...     pass
    """

    def __init__(self, detect_func, detect_smallest=1,
                 detect_min_size=0.,
                 detect_every=0.,
                 track_min_confidence=10., track_min_overlap_ratio=0.3,
                 track_max_gap=0.):

        super(TrackingByDetection, self).__init__()

        self.detect_func = detect_func
        self.detect_smallest = detect_smallest
        self.detect_min_size = detect_min_size
        self.detect_every = detect_every

        self.track_min_confidence = track_min_confidence
        self.track_min_overlap_ratio = track_min_overlap_ratio
        self.track_max_gap = track_max_gap

        self._hungarian = Munkres()

    def _kill_tracker(self, identifier):
        """Kill specific tracker"""
        del self._trackers[identifier]
        del self._confidences[identifier]
        del self._previous[identifier]

    def _match(self, rectangle1, rectangle2):
        overlap = rectangle1.intersect(rectangle2).area()
        if ((overlap < self.track_min_overlap_ratio * rectangle1.area()) or
            (overlap < self.track_min_overlap_ratio * rectangle2.area())):
            overlap = 0.
        return overlap

    def _associate(self, trackers, detections):
        """Associate trackers and detections with Hungarian algorithm

        Parameters
        ----------
        trackers : dict
            Dictionary where values are current trackers
            and keys are trackers identifiers.
        detections : list
            List of detections

        Returns
        -------
        match : dict
            Dictionary where values are trackers
            and keys are matched detection indices.
        """

        n_trackers, n_detections = len(trackers), len(detections)

        if n_trackers < 1 or n_detections < 1:
            return dict()

        n = max(n_trackers, n_detections)
        overlap_area = np.zeros((n, n))

        # list of (identifier, tracker) tuple
        trackers_ = trackers.items()
        for t, (identifier, tracker) in enumerate(trackers_):
            position = tracker.get_position()
            for d, detection in enumerate(detections):
                rectangle = dlib.drectangle(*detection)
                overlap_area[t, d] = self._match(position, rectangle)

        # find the best one-to-one mapping
        match = {}
        mapping = self._hungarian.compute(np.max(overlap_area) - overlap_area)
        for t, d in mapping:

            if t >= n_trackers or d >= n_detections:
                continue

            if overlap_area[t, d] > 0.:
                identifier, _ = trackers_[t]
                match[d] = identifier

        return match

    def _track(self, direction=FORWARD):
        """Actual tracking based on existing detections"""

        if direction == FORWARD:
            frame_cache = self._frame_cache
        elif direction == BACKWARD:
            frame_cache = reversed(self._frame_cache)
        else:
            raise NotImplementedError()

        self._trackers = {}
        self._confidences = {}
        self._previous = {}
        new_identifier = 0

        for t, frame in frame_cache:

            # update trackers & end those with low confidence
            for identifier, tracker in list(self._trackers.items()):
                confidence = tracker.update(frame)
                self._confidences[identifier] = confidence
                if confidence < self.track_min_confidence:
                    self._kill_tracker(identifier)

            # match trackers with detections at time t
            detections = [d for _, d, status in self._tracking_graph[t]
                          if status == DETECTION]
            match = self._associate(self._trackers, detections)

            # process all matched trackers
            for d, identifier in match.items():

                # connect the previous position of the tracker
                # to the (current) associated detection
                current = (t, detections[d], DETECTION)
                self._tracking_graph.add_edge(
                    self._previous[identifier], current,
                    confidence=self._confidences[identifier])

                # end the tracker
                self._kill_tracker(identifier)

            # process all unmatched trackers
            for identifier, tracker in self._trackers.items():

                # connect the previous position of the tracker
                # to the current position of the tracker
                position = tracker.get_position()
                position = (
                    position.left(),
                    position.top(),
                    position.right(),
                    position.bottom()
                )
                current = (t, position, direction)
                self._tracking_graph.add_edge(
                    self._previous[identifier], current,
                    confidence=self._confidences[identifier])

                # save current position of the tracker for next iteration
                self._previous[identifier] = current

            # start new trackers for all detections
            for d, detection in enumerate(detections):

                # start new tracker
                new_tracker = dlib.correlation_tracker()
                new_tracker.start_track(frame, dlib.drectangle(*detection))
                self._trackers[new_identifier] = new_tracker

                # save previous (t, position, status) tuple
                current = (t, detection, DETECTION)
                self._previous[new_identifier] = current

                # increment tracker identifier
                new_identifier = new_identifier + 1

    def _fix(self, track):
        """Fix track by merging matching forward/backward tracklets"""

        fixed_track = []
        for t, group in itertools.groupby(sorted(track), key=lambda x: x[0]):

            group = list(group)

            # make sure all positions are overlap enough
            error = False
            for (_, pos1, _), (_, pos2, _) in itertools.combinations(group, 2):

                rectangle1 = dlib.drectangle(*pos1)
                rectangle2 = dlib.drectangle(*pos2)

                if self._match(rectangle1, rectangle2) == 0:
                    error = True
                    break

            # status
            status = "+".join(
                sorted((status for _, _, status in group),
                       key=lambda s: {DETECTION: 2,
                                      FORWARD: 1,
                                      BACKWARD: 3}[s]))
            if error:
                status = "error({0})".format(status)

            # average position
            pos = tuple(int(round(v))
                        for v in np.mean(np.vstack([p for _, p, _ in group]),
                                         axis=0))

            fixed_track.append((t, pos, status))

        return fixed_track

    def _fill_gaps(self, tracks):

        # sort tracks by start and end timestamps
        tracks = sorted(tracks, key=get_min_max_t)

        # build graph where nodes are tracks and where matching tracks
        # less than "track_max_gap" away are connected
        graph = nx.Graph()
        for i in xrange(len(tracks)):
            graph.add_node(i)

        for i, j in itertools.combinations(xrange(len(tracks)), 2):

            # only try to match tracks with a short gap between them
            ti = tracks[i][-1][0]
            tj = tracks[j][0][0]
            if (tj < ti) or (tj - ti > self.track_max_gap):
                continue

            # match tracks whose last and first position match
            rectangle1 = dlib.drectangle(*tracks[i][-1][1])
            rectangle2 = dlib.drectangle(*tracks[j][0][1])
            if self._match(rectangle1, rectangle2):
                graph.add_edge(i, j)

        # merge tracks that are in the same connected component
        merged_tracks = []
        for group in nx.connected_components(graph):
            track = [item for t in sorted(group) for item in tracks[t]]
            merged_tracks.append(track)

        return merged_tracks

    def _forward_backward(self):

        # forward tracking
        self._track(direction=FORWARD)

        # backward tracking
        self._track(direction=BACKWARD)

        # remove timestamps
        timestamps = [t for t in self._tracking_graph
                      if not isinstance(t, tuple)]

        self._tracking_graph.remove_nodes_from(timestamps)

        # tracks are connected components in tracking graph
        tracks = nx.connected_components(
            self._tracking_graph.to_undirected(reciprocal=False))

        # merge matching backward/forward tracks
        tracks = [self._fix(track) for track in tracks]

        # fill gaps
        tracks = self._fill_gaps(tracks)

        # sort tracks by start and end timestamps
        for track in sorted(tracks, key=get_min_max_t):
            yield track

    def _reset(self):
        """Reset tracking"""
        self._frame_cache = []
        self._tracking_graph = nx.DiGraph()

    def _normalize_track(self, track, frame_width, frame_height):
        normalized_track = []
        for (t, (left, top, right, bottom), status) in track:
            left = left / frame_width
            right = right / frame_width
            top = top / frame_height
            bottom = bottom / frame_height
            normalized_track.append((t, (left, top, right, bottom), status))
        return normalized_track

    def __call__(self, video, segmentation):
        """
        Parameters
        ----------
        video : Video
        segmentation :
        """

        # should detection be applied to every frame or once every "x" frames?
        if self.detect_every > 0.0:
            every_x_frames = int(self.detect_every * video.frame_rate)
        else:
            every_x_frames = 1

        # estimate downscaling ratio
        width, height = video.size
        ratio = 1.0
        if self.detect_min_size > 0.0:
            ratio = self.detect_smallest / (self.detect_min_size * height)
            ratio = min(1.0, ratio)

        # tell video instance how to downscale its frames
        # (and keep track of previous setting)
        old_frame_width, old_frame_height = video.frame_size
        frame_width = int(width * ratio)
        frame_height = int(height * ratio)
        video.frame_size = (frame_width, frame_height)

        segment_generator = get_segment_generator(segmentation)
        segment_generator.send(None)
        self._reset()

        for i, (t, frame) in enumerate(video):

            segment = segment_generator.send(t)

            if segment:

                # forward/backward tracking
                for track in self._forward_backward():
                    yield self._normalize_track(track, frame_width, frame_height)

                # start fresh for next segment
                self._reset()

            # cache frame (for faster tracking)
            self._frame_cache.append((t, frame))

            self._tracking_graph.add_node(t)

            # apply detection every x frames
            if i % every_x_frames == 0:
                for detection in self.detect_func(frame):
                    self._tracking_graph.add_edge(t, (t, detection, DETECTION))

        for track in self._forward_backward():
            yield self._normalize_track(track, frame_width, frame_height)

        # revert frame size to its original setting
        if self.detect_min_size > 0.0:
            video.frame_size = (old_frame_width, old_frame_height)
