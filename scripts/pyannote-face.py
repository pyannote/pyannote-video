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
# Hervé BREDIN - http://herve.niderb.fr

"""Face detection and tracking

The standard pipeline is the following

      face tracking => feature extraction => face clustering

Usage:
  pyannote-face track [options] <video> <shot.json> <tracking>
  pyannote-face extract [options] <video> <tracking> <landmark_model> <embedding_model> <output>
  pyannote-face demo [options] <video> <precomputed> <output>
  pyannote-face (-h | --help)
  pyannote-face --version

General options:

  --ffmpeg=<ffmpeg>         Specify which `ffmpeg` to use.
  -h --help                 Show this screen.
  --version                 Show version.
  --verbose                 Show processing progress.

Face tracking options (track):

  <video>                   Path to video file.
  <shot.json>               Path to shot segmentation result file.
  <tracking>                Path to tracking result file.

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

Feature extraction options (features):

  <video>                   Path to video file.
  <tracking>                Path to tracking result file.
  <landmark_model>          Path to dlib facial landmark detection model.
  <embedding_model>         Path to dlib feature extraction model.
  <output>                  Path to features result file.

Visualization options (demo):

  <video>                   Path to video file.
  <precomputed>             Path to the precomputed features file.
  <output>                  Path to demo video file.

  --height=<pixels>         Height of demo video file [default: 400].
  --from=<sec>              Encode demo from <sec> seconds [default: 0].
  --until=<sec>             Encode demo until <sec> seconds.
  --shift=<sec>             Shift result files by <sec> seconds [default: 0].
  --yield_landmarks         Show landmarks in output video.
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

import numpy as np
import cv2

import dlib

LANDMARKS_DIM=(68,2)
EMBEDDING_DIM=128
MIN_OVERLAP_RATIO = 0.5
MIN_CONFIDENCE = 10.
MAX_GAP = 1.

TRACK_DTYPE=[
    ('time', 'float64'),
    ('track', 'int64'),
    ('bbox', 'float64', (4,)),
    ('status', '<U21'),
]

def bbox_to_rectangle(bbox,frame_width,frame_height, double=True):
    rectangle = dlib.drectangle if double else dlib.rectangle
    left, top, right, bottom=bbox
    left = int(left * frame_width)
    right = int(right * frame_width)
    top = int(top * frame_height)
    bottom = int(bottom * frame_height)
    face = rectangle(left, top, right, bottom)
    return face
def rectangle_to_bbox(rectangle,frame_width,frame_height):
    left, top, right, bottom=rectangle.left(),rectangle.top(),rectangle.right(),rectangle.bottom()
    left/=frame_width
    top/=frame_height
    right/=frame_width
    bottom/=frame_height
    return (left, top, right, bottom)



def getGenerator(precomputed, frame_width, frame_height,yield_landmarks=False, double=True):
    """Parse precomputed face file and generate timestamped faces along with landmarks if yield_landmarks
    Parameters:
    -----------
    precomputed: path to the precomputed file (np array, described in README)
    frame_width, frame_height: self-explanatory
    yield_landmarks: if True : yields landmarks, else, yields only bounding boxes
        Defaults to False.
    """

    # load precomputed file and sort it by timestamp
    precomputed = np.load(precomputed)
    precomputed.sort(order='time')

    # t is the time sent by the frame generator
    t = yield

    faces = []
    currentT = None

    for face in precomputed:
        T, identifier, bbox, status=face['time'],face['track'],face['bbox'],face['status']
        if yield_landmarks:
            landmarks=face['landmarks']
            landmarks[:, 0] = np.round(landmarks[:, 0] * frame_width)
            landmarks[:, 1] = np.round(landmarks[:, 1] * frame_height)
        face=bbox_to_rectangle(bbox,frame_width, frame_height,double)

        # load all faces from current frame and only those faces
        if T == currentT or currentT is None:
            if yield_landmarks:
                faces.append((identifier, face, status,landmarks))
            else:
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
            if yield_landmarks:
                faces = [(identifier, face, status,landmarks)]
            else:
                faces = [(identifier, face, status)]
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
    save_tracks=[]
    for identifier, track in enumerate(tracking(video, shot)):
        for i, (t, (left, top, right, bottom), status) in enumerate(track):
            save_tracks.append((
                t,
                identifier,
                (left, top, right, bottom),
                status
            ))
    tracks=np.array(
        save_tracks,
        dtype=TRACK_DTYPE
    )
    np.save(output,tracks)

def extract(video, landmark_model, embedding_model, tracking, output):
    """Facial features detection"""

    # face generator
    frame_width, frame_height = video.frame_size
    faceGenerator = getGenerator(tracking,
                                     frame_width, frame_height,
                                     yield_landmarks=False,
                                     double=False)
    faceGenerator.send(None)
    face = Face(landmarks=landmark_model,
                embedding=embedding_model)

    save_extracted=[]
    for timestamp, rgb in video:
        # get all detected faces at this time
        T, faces = faceGenerator.send(timestamp)
        # not that T might be differ slightly from t
        # due to different steps in frame iteration
        for identifier, bounding_box, status in faces:

            landmarks = face.get_landmarks(rgb, bounding_box)
            embedding = face.get_embedding(rgb, landmarks)
            save_landmarks=[]
            for p in landmarks.parts():
                x, y = p.x, p.y
                save_landmarks.append((x / frame_width, y / frame_height))
            save_embedding=[]
            for x in embedding:
                save_embedding.append(x)
            (left, top, right, bottom)=rectangle_to_bbox(bounding_box,frame_width, frame_height)
            save_extracted.append((
                T,
                identifier,
                (left, top, right, bottom),
                status,
                save_landmarks,
                save_embedding
            ))
    extracted=np.array(
        save_extracted,
        dtype=TRACK_DTYPE+[
          ('landmarks', 'float64', LANDMARKS_DIM),
          ('embeddings', 'float64', (EMBEDDING_DIM,))
        ]
    )
    np.save(output,extracted)

def get_make_frame(video, precomputed,yield_landmarks=False, labels=None,
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


    generator=getGenerator(precomputed, width, height,
        yield_landmarks=yield_landmarks, double=True)
    generator.send(None)

    if labels is None:
        labels = dict()

    def make_frame(t):

        frame = video(t)
        _,faces = generator.send(t - shift)

        cv2.putText(frame, '{t:.3f}'.format(t=t), (10, height-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1, 8, False)
        for i, element in enumerate(faces):
            if yield_landmarks:
                identifier, face, status,landmarks=element
            else:
                identifier, face, status=element
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
            if yield_landmarks:
                pt1 = (int(landmarks[27][0]), int(landmarks[27][1]))
                pt2 = (int(landmarks[33][0]), int(landmarks[33][1]))
                cv2.line(frame, pt1, pt2, color, 1)

        return frame

    return make_frame


def demo(filename, precomputed, output, t_start=0., t_end=None, shift=0.,
         labels=None, yield_landmarks=False, height=200, ffmpeg=None):

    # parse label file
    if labels is not None:
        with open(labels, 'r') as f:
            labels = {}
            for line in f:
                identifier, label = line.strip().split()
                identifier = int(identifier)
                labels[identifier] = label

    video = Video(filename, ffmpeg=ffmpeg)

    from moviepy.editor import VideoClip, AudioFileClip

    make_frame = get_make_frame(video, precomputed, yield_landmarks=yield_landmarks,
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
    ffmpeg = arguments['--ffmpeg']

    verbose = arguments['--verbose']

    video = Video(filename, ffmpeg=ffmpeg, verbose=verbose)

    # face tracking
    if arguments['track']:

        shot = arguments['<shot.json>']
        tracking = arguments['<tracking>']
        detect_min_size = float(arguments['--min-size'])
        detect_every = float(arguments['--every'])
        track_min_overlap_ratio = float(arguments['--min-overlap'])
        track_min_confidence = float(arguments['--min-confidence'])
        track_max_gap = float(arguments['--max-gap'])
        track(video, shot, tracking,
              detect_min_size=detect_min_size,
              detect_every=detect_every,
              track_min_overlap_ratio=track_min_overlap_ratio,
              track_min_confidence=track_min_confidence,
              track_max_gap=track_max_gap)

    # facial features detection
    if arguments['extract']:

        tracking = arguments['<tracking>']
        landmark_model = arguments['<landmark_model>']
        embedding_model = arguments['<embedding_model>']
        output = arguments['<output>']
        extract(video, landmark_model, embedding_model, tracking, output)


    if arguments['demo']:

        precomputed = arguments['<precomputed>']
        output = arguments['<output>']

        t_start = float(arguments['--from'])
        t_end = arguments['--until']
        t_end = float(t_end) if t_end else None

        shift = float(arguments['--shift'])

        labels = arguments['--label']
        if not labels:
            labels = None

        yield_landmarks = arguments['--yield_landmarks']

        height = int(arguments['--height'])

        demo(filename, precomputed, output,
             t_start=t_start, t_end=t_end,
             yield_landmarks=yield_landmarks, height=height,
             shift=shift, labels=labels, ffmpeg=ffmpeg)
