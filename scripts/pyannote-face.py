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

"""Face detection and tracking

The standard pipeline is the following

      face tracking => facial landmarks detection => feature extraction

Usage:
  pyannote-face track [options] <video> <shot.json> <output>
  pyannote-face landmarks [options] <video> <model> <tracking> <output>
  pyannote-face features [options] <video> <model> <landmark> <output>
  pyannote-face demo [options] <video> <tracking> <output>
  pyannote-face (-h | --help)
  pyannote-face --version

General options:

  -h --help                 Show this screen.
  --version                 Show version.
  --verbose                 Show processing progress.

Face tracking options (track):

  <video>                   Path to video file.
  <shot.json>               Path to shot segmentation result file.
  <output>                  Path to tracking result file.

  --min-size=<ratio>        Approximate size (in video height ratio) of the
                            smallest face that should be detected. Default is
                            to try and detect any object [default: 0.0].
  --every=<seconds>         Only apply detection every <seconds> seconds.
                            Default is to process every frame [default: 0.0].
  --min-overlap=<ratio>     Associates face with tracker if overlap is greater
                            than <ratio> [default: 0.5].
  --min-confidence=<float>  Reset trackers with confidence lower than <float>
                            [default: 10.].
  --max-gap=<float>         Bridge gaps with duration shorter than <float>
                            [default: 1.].

Facial landmarks detection options (landmarks):

  <video>                   Path to video file.
  <model>                   Path to dlib facial landmark detection model.
  <tracking>                Path to tracking result file.
  <output>                  Path to facial landmarks detection result file.

Openface feature extraction options (features):

  <video>                   Path to video file.
  <model>                   Path to Openface feature extraction model.
  <landmarks>               Path to facial landmarks detection result file.
  <output>                  Path to facial landmarks detection result file.
  --torch=<path>            Path to torch [default: 'th'].

Visualization options (demo):

  <video>                   Path to video file.
  <tracking>                Path to tracking result file.
  <output>                  Path to demo video file.

  --height=<pixels>         Height of demo video file [default: 400].
  --from=<sec>              Encode demo from <sec> seconds [default: 0].
  --until=<sec>             Encode demo until <sec> seconds.
  --shift=<sec>             Shift result files by <sec> seconds [default: 0].
  --landmark=<path>         Path to facial landmarks detection result file.
  --label=<path>            Path to track identification result file.

"""

from __future__ import division

from docopt import docopt

from pyannote.core import Annotation
import pyannote.core.json

from pyannote.video import __version__
from pyannote.video import Video
from pyannote.video import Face
from pyannote.video import FaceTracking

from pandas import read_table

from six.moves import zip
import numpy as np
import cv2

import dlib


MIN_OVERLAP_RATIO = 0.5
MIN_CONFIDENCE = 10.
MAX_GAP = 1.

FACE_TEMPLATE = ('{t:.3f} {identifier:d} '
                 '{left:.3f} {top:.3f} {right:.3f} {bottom:.3f} '
                 '{status:s}\n')


def getFaceGenerator(tracking, frame_width, frame_height, double=True):
    """Parse precomputed face file and generate timestamped faces"""

    # load tracking file and sort it by timestamp
    names = ['t', 'track', 'left', 'top', 'right', 'bottom', 'status']
    dtype = {'left': np.float32, 'top': np.float32,
             'right': np.float32, 'bottom': np.float32}
    tracking = read_table(tracking, delim_whitespace=True, header=None,
                          names=names, dtype=dtype)
    tracking = tracking.sort_values('t')

    # t is the time sent by the frame generator
    t = yield

    rectangle = dlib.drectangle if double else dlib.rectangle

    faces = []
    currentT = None

    for _, (T, identifier, left, top, right, bottom, status) in tracking.iterrows():

        left = int(left * frame_width)
        right = int(right * frame_width)
        top = int(top * frame_height)
        bottom = int(bottom * frame_height)

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


def getLandmarkGenerator(shape, frame_width, frame_height):
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
            [coordinate for coordinate in row[2:]])))
        landmarks[:, 0] = np.round(landmarks[:, 0] * frame_width)
        landmarks[:, 1] = np.round(landmarks[:, 1] * frame_height)

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


def track(video, shot, output,
          detect_min_size=0.0,
          detect_every=0.0,
          track_min_overlap_ratio=MIN_OVERLAP_RATIO,
          track_min_confidence=MIN_CONFIDENCE,
          track_max_gap=MAX_GAP):
    """Tracking by detection"""

    tracking = FaceTracking(detect_min_size=detect_min_size,
                            detect_every=detect_every,
                            track_min_overlap_ratio=track_min_overlap_ratio,
                            track_min_confidence=track_min_confidence,
                            track_max_gap=track_max_gap)

    with open(shot, 'r') as fp:
        shot = pyannote.core.json.load(fp)

    if isinstance(shot, Annotation):
        shot = shot.get_timeline()

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
    frame_width, frame_height = video.frame_size
    faceGenerator = getFaceGenerator(tracking,
                                     frame_width, frame_height,
                                     double=False)
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
                    foutput.write(' {x:.5f} {y:.5f}'.format(x=x / frame_width,
                                                            y=y / frame_height))
                foutput.write('\n')

            foutput.flush()

def features(video, model, shape, output, torch='th'):
    """Openface FaceNet feature extraction"""

    face = Face(size=96, openface=model, torch=torch)

    # shape generator
    frame_width, frame_height = video.frame_size
    landmarkGenerator = getLandmarkGenerator(shape, frame_width, frame_height)
    landmarkGenerator.send(None)

    with open(output, 'w') as foutput:

        for timestamp, rgb in video:

            T, shapes = landmarkGenerator.send(timestamp)

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

def get_make_frame(video, tracking, landmark=None, labels=None,
                   height=200, shift=0.0):

    COLORS = [
        (240, 163, 255), (  0, 117, 220), (153,  63,   0), ( 76,   0,  92),
        ( 25,  25,  25), (  0,  92,  49), ( 43, 206,  72), (255, 204, 153),
        (128, 128, 128), (148, 255, 181), (143, 124,   0), (157, 204,   0),
        (194,   0, 136), (  0,  51, 128), (255, 164,   5), (255, 168, 187),
        ( 66, 102,   0), (255,   0,  16), ( 94, 241, 242), (  0, 153, 143),
        (224, 255, 102), (116,  10, 255), (153,   0,   0), (255, 255, 128),
        (255, 255,   0), (255,  80,   5)
    ]

    video_width, video_height = video.size
    ratio = height / video_height
    width = int(ratio * video_width)
    video.frame_size = (width, height)

    faceGenerator = getFaceGenerator(tracking, width, height, double=True)
    faceGenerator.send(None)

    if landmark:
        landmarkGenerator = getLandmarkGenerator(landmark, width, height)
        landmarkGenerator.send(None)

    if labels is None:
        labels = dict()

    def make_frame(t):

        frame = video(t)
        _, faces = faceGenerator.send(t - shift)

        if landmark:
            _, landmarks = landmarkGenerator.send(t - shift)

        cv2.putText(frame, '{t:.3f}'.format(t=t), (10, height-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1, 8, False)
        for i, (identifier, face, _) in enumerate(faces):
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

            # Draw nose
            if landmark:
                _, points = landmarks[i]
                pt1 = (int(points[27, 0]), int(points[27, 1]))
                pt2 = (int(points[33, 0]), int(points[33, 1]))
                cv2.line(frame, pt1, pt2, color, 1)

        return frame

    return make_frame


def demo(filename, tracking, output, t_start=0., t_end=None, shift=0.,
         labels=None, landmark=None, height=200):

    # parse label file
    if labels is not None:
        with open(labels, 'r') as f:
            labels = {}
            for line in f:
                identifier, label = line.strip().split()
                identifier = int(identifier)
                labels[identifier] = label

    video = Video(filename)

    import os
    os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'
    # from moviepy.video.io.VideoFileClip import VideoFileClip

    from moviepy.editor import VideoClip, AudioFileClip

    make_frame = get_make_frame(video, tracking, landmark=landmark,
                                labels=labels, height=height, shift=shift)
    video_clip = VideoClip(make_frame, duration=video.duration)
    audio_clip = AudioFileClip(filename)
    clip = video_clip.set_audio(audio_clip)

    if t_end is None:
        t_end = video.duration

    clip.subclip(t_start, t_end).write_videofile(output, fps=video.frame_rate)

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
        detect_min_size = float(arguments['--min-size'])
        detect_every = float(arguments['--every'])
        track_min_overlap_ratio = float(arguments['--min-overlap'])
        track_min_confidence = float(arguments['--min-confidence'])
        track_max_gap = float(arguments['--max-gap'])
        track(video, shot, output,
              detect_min_size=detect_min_size,
              detect_every=detect_every,
              track_min_overlap_ratio=track_min_overlap_ratio,
              track_min_confidence=track_min_confidence,
              track_max_gap=track_max_gap)

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
        torch = arguments['--torch']
        features(video, model, shape, output, torch=torch)

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

        landmark = arguments['--landmark']
        if not landmark:
            landmark = None

        height = int(arguments['--height'])

        demo(filename, tracking, output,
             t_start=t_start, t_end=t_end,
             landmark=landmark, height=height,
             shift=shift, labels=labels)
