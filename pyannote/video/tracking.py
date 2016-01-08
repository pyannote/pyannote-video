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

"""Tracking by detection"""

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
    min_confidence : float, optional
        Kill trackers whose confidence goes below this value. Defaults to 10.
    min_overlap_ratio : float, optional
        Do not associate trackers and detections if their overlap ratio goes
        below this value. Defaults to 0.5.

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

    def __init__(self, detect_func, min_confidence=10., min_overlap_ratio=0.5):

        super(TrackingByDetection, self).__init__()

        self.detect_func = detect_func
        self.min_confidence = min_confidence
        self.min_overlap_ratio = min_overlap_ratio

        self._hungarian = Munkres()

    def _reset(self):
        """Reset tracking"""
        self._frame_cache = []
        self._tracking_graph = nx.DiGraph()

    def _kill_tracker(self, identifier):
        """Kill specific tracker"""
        del self._trackers[identifier]
        del self._confidences[identifier]
        del self._previous[identifier]

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
                overlap_area[t, d] = position.intersect(rectangle).area()

        # find the best one-to-one mapping
        mapping = self._hungarian.compute(np.max(overlap_area) - overlap_area)
        match = {}

        for t, d in mapping:

            if t >= n_trackers or d >= n_detections:
                continue

            area = overlap_area[t, d]

            detection = dlib.drectangle(*detections[d])
            detection_area = detection.area()

            identifier, tracker = trackers_[t]
            tracker_area = tracker.get_position().area()

            if ((area > detection_area * self.min_overlap_ratio) or
                (area > tracker_area * self.min_overlap_ratio)):
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
                if confidence < self.min_confidence:
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
                    int(round(position.left())),
                    int(round(position.top())),
                    int(round(position.right())),
                    int(round(position.bottom()))
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

    def _fix_track(self, track):

        fixed_track = []
        for t, group in itertools.groupby(sorted(track), key=lambda x: x[0]):

            group = list(group)

            # make sure all positions are overlap enough
            error = False
            for (_, pos1, _), (_, pos2, _) in itertools.combinations(group, 2):
                pos1 = dlib.drectangle(*pos1)
                pos2 = dlib.drectangle(*pos2)
                overlap = pos1.intersect(pos2).area()

                if ((overlap < pos1.area() * self.min_overlap_ratio) or
                   (overlap < pos2.area() * self.min_overlap_ratio)):
                    error = True
                    break

            # status
            status = "-".join(sorted(status for _, _, status in group))
            if error:
                status = "+".join([ERROR, status])

            # average position
            pos = tuple(int(round(v))
                        for v in np.mean(np.vstack([p for _, p, _ in group]),
                                         axis=0))

            fixed_track.append((t, pos, status))

        return fixed_track

    def _forward_backward(self):

        # forward tracking
        self._track(direction=FORWARD)

        # backward tracking
        self._track(direction=BACKWARD)

        # remove timestamps
        timestamps = [t for t in self._tracking_graph
                      if not isinstance(t, tuple)]

        self._tracking_graph.remove_nodes_from(timestamps)

        # tracks sorted by start and end timestamps
        tracks = nx.connected_components(
            self._tracking_graph.to_undirected(reciprocal=False))
        tracks = sorted(tracks, key=get_min_max_t)

        for track in tracks:
            yield self._fix_track(track)

    def __call__(self, video, segmentation):
        """
        Parameters
        ----------
        video : Video
        segmentation :
        """

        segment_generator = get_segment_generator(segmentation)
        segment_generator.send(None)
        self._reset()

        for t, frame in video:

            segment = segment_generator.send(t)

            if segment:

                # forward/backward tracking
                for track in self._forward_backward():
                    yield track

                # start fresh for next segment
                self._reset()

            # cache frame (for faster tracking)
            self._frame_cache.append((t, frame))

            # detection
            self._tracking_graph.add_node(t)
            for detection in self.detect_func(frame):
                self._tracking_graph.add_edge(t, (t, detection, DETECTION))

        for track in self._forward_backward():
            yield track
