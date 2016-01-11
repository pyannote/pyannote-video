#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2015 CNRS

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

"""Face detection and tracking

The standard pipeline is the following (with optional face tracking)

face detection => (face tracking =>) landmarks detection => feature extraction

Usage:
  pyannote-face track [options] [--verbose] <video> <shot.json> <output>
  pyannote-face landmarks [--verbose] <video> <model> <tracking> <output>
  pyannote-face features [--verbose] <video> <model> <landmark> <output>
  pyannote-face demo [--from=<sec>] [--until=<sec>] [--shift=<sec>] [--label=<path>] <video> <tracking> <output>
  pyannote-face (-h | --help)
  pyannote-face --version

Options:
  --smallest=<size>         (Approximate) size of smallest face [default: 36].
  --min-overlap=<ratio>     Associates face with tracker if overlap is greater
                            than <ratio> [default: 0.5].
  --min-confidence=<float>  Reset trackers with confidence lower than <float>
                            [default: 10.].
  --max-gap=<float>         Bridge gaps with duration shorter than <float>
                            [default: 1.].
  --from=<sec>              Encode demo from <sec> seconds [default: 0].
  --until=<sec>             Encode demo until <sec> seconds.
  --shift=<sec>             Shift tracks by <sec> seconds [default: 0].
  --label=<path>            Track labels.
  -h --help                 Show this screen.
  --version                 Show version.
  --verbose                 Show progress.
"""

from docopt import docopt

from pyannote.core.json import load
from pyannote.video import __version__
from pyannote.video import Video
from pyannote.video import Face
from pyannote.video import FaceTracking

from pandas import read_table

from six.moves import zip
import numpy as np
import cv2

import dlib


SMALLEST_DEFAULT = 36
MIN_OVERLAP_RATIO = 0.5
MIN_CONFIDENCE = 10.
MAX_GAP = 1.

FACE_TEMPLATE = ('{t:.3f} {identifier:d} '
                 '{left:d} {top:d} {right:d} {bottom:d} '
                 '{status:s}\n')


def getFaceGenerator(tracking, double=True):
    """Parse precomputed face file and generate timestamped faces"""

    # load tracking file and sort it by timestamp
    names = ['t', 'track', 'left', 'top', 'right', 'bottom', 'status']
    dtype = {'left': np.int32, 'top': np.int32,
             'right': np.int32, 'bottom': np.int32}
    tracking = read_table(tracking, delim_whitespace=True, header=None,
                          names=names, dtype=dtype)
    tracking = tracking.sort_values('t')

    # t is the time sent by the frame generator
    t = yield

    rectangle = dlib.drectangle if double else dlib.rectangle

    faces = []
    currentT = None

    for _, (T, identifier, left, top, right, bottom, status) in tracking.iterrows():

        face = rectangle(left, top, right, bottom)

        # load all faces from current frame and only those faces
        if T == currentT or currentT is None:
            faces.append((identifier, face, status))
            currentT = T
            continue

        # once all faces at current time are loaded
        # wait until t reaches current time
        # then returns all faces at once

        while True:

            # wait...
            if currentT > t:
                t = yield t, []
                continue

            # return all faces at once
            t = yield currentT, faces

            # reset current time and corresponding faces
            faces = [(identifier, face, status)]
            currentT = T
            break

    while True:
        t = yield t, []


def pairwise(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def getShapeGenerator(shape):
    """Parse precomputed shape file and generate timestamped shapes"""

    # load landmarks file
    shape = read_table(shape, delim_whitespace=True, header=None)

    # deduce number of landmarks from file dimension
    _, d = shape.shape
    n_points = (d - 2) / 2

    # t is the time sent by the frame generator
    t = yield

    shapes = []
    currentT = None

    for _, row in shape.iterrows():

        T = float(row[0])
        identifier = int(row[1])
        landmarks = np.float32(list(pairwise(
            [int(coordinate) for coordinate in row[2:]])))

        # load all shapes from current frame
        # and only those shapes
        if T == currentT or currentT is None:
            shapes.append((identifier, landmarks))
            currentT = T
            continue

        # once all shapes at current time are loaded
        # wait until t reaches current time
        # then returns all shapes at once

        while True:

            # wait...
            if currentT > t:
                t = yield t, []
                continue

            # return all shapes at once
            t = yield currentT, shapes

            # reset current time and corresponding shapes
            shapes = [(identifier, landmarks)]
            currentT = T
            break

    while True:
        t = yield t, []


def track(video, shot, output, smallest=36,
          min_overlap_ratio=MIN_OVERLAP_RATIO, min_confidence=MIN_CONFIDENCE,
          max_gap=MAX_GAP):
    """Tracking by detection"""

    tracking = FaceTracking(smallest=smallest,
                            min_overlap_ratio=min_overlap_ratio,
                            min_confidence=min_confidence,
                            max_gap=max_gap)

    shot = load(shot)

    with open(output, 'w') as foutput:

        for identifier, track in enumerate(tracking(video, shot)):

            for t, (left, top, right, bottom), status in track:

                foutput.write(FACE_TEMPLATE.format(
                    t=t, identifier=identifier, status=status,
                    left=left, right=right, top=top, bottom=bottom))

            foutput.flush()

def landmark(video, model, tracking, output):
    """Facial features detection"""

    # face generator
    faceGenerator = getFaceGenerator(tracking, double=False)
    faceGenerator.send(None)

    face = Face(landmarks=model)

    with open(output, 'w') as foutput:

        for timestamp, rgb in video:

            # get all detected faces at this time
            T, faces = faceGenerator.send(timestamp)
            # not that T might be differ slightly from t
            # due to different steps in frame iteration

            for identifier, boundingBox, _ in faces:

                landmarks = face._get_landmarks(rgb, boundingBox)

                foutput.write('{t:.3f} {identifier:d}'.format(
                    t=T, identifier=identifier))
                for x, y in landmarks:
                    foutput.write(' {x:d} {y:d}'.format(x=int(x), y=int(y)))
                foutput.write('\n')

            foutput.flush()

def features(video, model, shape, output):
    """Openface FaceNet feature extraction"""

    face = Face(size=96, openface=model)

    # shape generator
    shapeGenerator = getShapeGenerator(shape)
    shapeGenerator.send(None)

    with open(output, 'w') as foutput:

        for timestamp, rgb in video:

            T, shapes = shapeGenerator.send(timestamp)

            for identifier, landmarks in shapes:
                normalized_rgb = face._get_normalized(rgb, landmarks)
                normalized_bgr = cv2.cvtColor(normalized_rgb,
                                              cv2.COLOR_BGR2RGB)
                openface = face._get_openface(normalized_bgr)

                foutput.write('{t:.3f} {identifier:d}'.format(
                    t=T, identifier=identifier))
                for x in openface:
                    foutput.write(' {x:.5f}'.format(x=x))
                foutput.write('\n')

            foutput.flush()

def get_fl(tracking, shift=0., labels=None):

    COLORS = [
        (240, 163, 255), (  0, 117, 220), (153,  63,   0), ( 76,   0,  92),
        ( 25,  25,  25), (  0,  92,  49), ( 43, 206,  72), (255, 204, 153),
        (128, 128, 128), (148, 255, 181), (143, 124,   0), (157, 204,   0),
        (194,   0, 136), (  0,  51, 128), (255, 164,   5), (255, 168, 187),
        ( 66, 102,   0), (255,   0,  16), ( 94, 241, 242), (  0, 153, 143),
        (224, 255, 102), (116,  10, 255), (153,   0,   0), (255, 255, 128),
        (255, 255,   0), (255,  80,   5)
    ]

    faceGenerator = getFaceGenerator(tracking, double=True)
    faceGenerator.send(None)

    if labels is None:
        labels = dict()

    def overlay(get_frame, timestamp):
        frame = get_frame(timestamp)
        height, width, _ = frame.shape
        _, faces = faceGenerator.send(timestamp - shift)

        cv2.putText(frame, '{t:.3f}'.format(t=timestamp), (10, height-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1, 8, False)
        for identifier, face, _ in faces:
            color = COLORS[identifier % len(COLORS)]

            # Draw face bounding box
            pt1 = (int(face.left()), int(face.top()))
            pt2 = (int(face.right()), int(face.bottom()))
            cv2.rectangle(frame, pt1, pt2, color, 2)

            # Print tracker identifier
            cv2.putText(frame, '#{identifier:d}'.format(identifier=identifier),
                        (pt1[0], pt2[1] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, 8, False)

            # Print track label
            label = labels.get(identifier, '')
            cv2.putText(frame,
                        '{label:s}'.format(label=label),
                        (pt1[0], pt1[1] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, 8, False)
        return frame

    return overlay


def demo(filename, tracking, output, t_start=0., t_end=None, shift=0.,
         labels=None):

    import os
    os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'
    from moviepy.video.io.VideoFileClip import VideoFileClip

    if labels is not None:
        with open(labels, 'r') as f:
            labels = {}
            for line in f:
                identifier, label = line.strip().split()
                identifier = int(identifier)
                labels[identifier] = label

    original_clip = VideoFileClip(filename)
    modified_clip = original_clip.fl(get_fl(tracking,
                                            shift=shift,
                                            labels=labels))
    cropped_clip = modified_clip.subclip(t_start=t_start, t_end=t_end)
    cropped_clip.write_videofile(output)


if __name__ == '__main__':

    # parse command line arguments
    version = 'pyannote-face {version}'.format(version=__version__)
    arguments = docopt(__doc__, version=version)

    # initialize video
    filename = arguments['<video>']

    verbose = arguments['--verbose']

    video = Video(filename, verbose=verbose)

    # face tracking
    if arguments['track']:

        shot = arguments['<shot.json>']
        output = arguments['<output>']
        min_overlap_ratio = float(arguments['--min-overlap'])
        min_confidence = float(arguments['--min-confidence'])
        max_gap = float(arguments['--max-gap'])
        track(video, shot, output,
              min_overlap_ratio=min_overlap_ratio,
              min_confidence=min_confidence,
              max_gap=max_gap)

    # facial features detection
    if arguments['landmarks']:

        tracking = arguments['<tracking>']
        model = arguments['<model>']
        output = arguments['<output>']
        landmark(video, model, tracking, output)

    # openface features extraction
    if arguments['features']:

        model = arguments['<model>']
        shape = arguments['<landmark>']
        output = arguments['<output>']
        features(video, model, shape, output)

    if arguments['demo']:

        tracking = arguments['<tracking>']
        output = arguments['<output>']

        t_start = float(arguments['--from'])
        t_end = arguments['--until']
        t_end = float(t_end) if t_end else None

        shift = float(arguments['--shift'])
        labels = arguments['--label']
        if not labels:
            labels = None

        demo(filename, tracking, output,
             t_start=t_start, t_end=t_end,
             shift=shift, labels=labels)
