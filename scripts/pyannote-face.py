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

 face detection => face tracking => landmark detection => feature extraction

Usage:
  pyannote-face detect [--verbose] [options] <video> <output>
  pyannote-face track [--verbose] <video> <shot> <detection> <output>
  pyannote-face landmark [--verbose] <video> <model> <tracking> <output>
  pyannote-face features [--verbose] <video> <model> <landmark> <output>
  pyannote-face demo [--from=<sec>] [--until=<sec>] <video> <tracking> <output>
  pyannote-face (-h | --help)
  pyannote-face --version

Options:
  --every=<msec>            Process one frame every <msec> milliseconds.
  --smallest=<size>         (Approximate) size of smallest face [default: 36].
  --from=<sec>              Encode demo from <sec> seconds [default: 0].
  --until=<sec>             Encode demo until <sec> seconds.
  --min-overlap=<ratio>     Associates face with tracker if overlap is greater
                            than <ratio> [default: 0.5].
  --min-confidence=<float>  Reset trackers with confidence lower than <float>
                            [default: 10.].
  -h --help                 Show this screen.
  --version                 Show version.
  --verbose                 Show progress.
"""

SMALLEST_DEFAULT = 36
MIN_OVERLAP_RATIO = 0.5
MIN_CONFIDENCE = 10.

from docopt import docopt
from pyannote.video import __version__
from pyannote.video.video import Video
from pyannote.video import Face

from six.moves import zip
from tqdm import tqdm
from munkres import Munkres
import numpy as np
import cv2

import dlib


FACE_TEMPLATE = ('{t:.3f} {identifier:d} '
                 '{left:d} {top:d} {right:d} {bottom:d} '
                 '{confidence:.3f}\n')


def getShotGenerator(shotFile):
    """Parse precomputed shot file and generate boundary timestamps"""

    t = yield

    with open(shotFile, 'r') as f:

        for line in f:
            T = float(line.strip())

            while True:
                # loop until a large enough t is sent to the generator
                if T > t:
                    t = yield
                    continue

                # else, we found a new shot
                t = yield T
                break


def getFaceGenerator(detection, double=True):
    """Parse precomputed face file and generate timestamped faces"""

    # t is the time sent by the frame generator
    t = yield

    rectangle = dlib.drectangle if double else dlib.rectangle

    with open(detection, 'r') as f:

        faces = []
        currentT = None

        for line in f:

            # parse line
            # time, identifier, left, top, right, bottom, confidence
            tokens = line.strip().split()
            T = float(tokens[0])
            identifier = int(tokens[1])
            face = rectangle(*[int(token) for token in tokens[2:6]])
            confidence = float(tokens[6])

            # load all faces from current frame
            # and only those faces
            if T == currentT or currentT is None:
                faces.append((identifier, face, confidence))
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
                faces = [(identifier, face, confidence)]
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

    # t is the time sent by the frame generator
    t = yield

    with open(shape, 'r') as f:

        shapes = []
        currentT = None

        for line in f:

            # parse line
            # time, identifier, x1, y1, ..., x68, y68
            tokens = line.strip().split()
            T = float(tokens[0])
            identifier = int(tokens[1])
            landmarks = np.float32(list(pairwise(
                [int(token) for token in tokens[2:]])))

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



def detect(video, output, step=None, upscale=1, show_progress=False):
    """Face detection"""

    # frame iterator
    generator = video.iterframes(step=step, with_time=True)
    if show_progress:

        if step is None:
            total = video.duration * video.fps
        else:
            total = video.duration / step

        generator = tqdm(iterable=generator,
                         total=total,
                         leave=True, mininterval=1.,
                         unit='frames', unit_scale=True)

    # face detector
    faceDetector = dlib.get_frontal_face_detector()

    identifier = 0

    with open(output, 'w') as foutput:

        for t, frame in generator:

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            for boundingBox in faceDetector(gray, upscale):

                foutput.write(FACE_TEMPLATE.format(
                    t=t, identifier=identifier, confidence=0.000,
                    left=boundingBox.left(), right=boundingBox.right(),
                    top=boundingBox.top(), bottom=boundingBox.bottom()))

                identifier = identifier + 1


def track(video, shot, detection, output,
          min_overlap_ratio=MIN_OVERLAP_RATIO,
          min_confidence=MIN_CONFIDENCE,
          show_progress=False):
    """Tracking by detection"""

    # frame generator
    frames = video.iterframes(with_time=True)
    if show_progress:
        frames = tqdm(iterable=frames,
                      total=video.duration * video.fps,
                      leave=True, mininterval=1.,
                      unit='frames', unit_scale=True)

    # shot generator
    shotGenerator = getShotGenerator(shot)
    shotGenerator.send(None)

    # face generator
    faceGenerator = getFaceGenerator(detection)
    faceGenerator.send(None)

    # Hungarian algorithm for face/tracker matching
    hungarian = Munkres()

    trackers = dict()
    confidences = dict()
    identifier = 0

    with open(output, 'w') as foutput:

        for timestamp, frame in frames:

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            shot = shotGenerator.send(timestamp)

            # reset trackers at shot boundaries
            if shot:
                trackers.clear()
                confidences.clear()

            # get all detected faces at this time
            T, faces = faceGenerator.send(timestamp)
            # not that T might be differ slightly from t
            # due to different steps in frame iteration

            # update all trackers and store their confidence
            for i, tracker in trackers.items():
                confidences[i] = tracker.update(gray)

            # reset trackers when it looses confidence
            for i, tracker in list(trackers.items()):
                if confidences[i] < min_confidence:
                    del trackers[i]
                    del confidences[i]

            # set of (yet) un-associated trackers
            unmatched = set(trackers)

            Nt, Nf = len(trackers), len(faces)
            if Nt and Nf:

                # compute intersection for every tracker/face pair
                N = max(Nt, Nf)
                areas = np.zeros((N, N))
                trackers_ = trackers.items()
                for t, (i, tracker) in enumerate(trackers_):
                    position = tracker.get_position()
                    for f, (_, face, _) in enumerate(faces):
                        areas[t, f] = position.intersect(face).area()

                # find the best one-to-one mapping
                mapping = hungarian.compute(np.max(areas) - areas)

                for t, f in mapping:

                    if t >= Nt or f >= Nf:
                        continue

                    area = areas[t, f]

                    _, face, _ = faces[f]
                    faceArea = face.area()

                    i, tracker = trackers_[t]
                    trackerArea = tracker.get_position().area()

                    # if enough overlap,
                    # re-intialize tracker and mark face as matched
                    if ((area > faceArea * min_overlap_ratio) or
                        (area > trackerArea * min_overlap_ratio)):
                        tracker.start_track(gray, face)
                        unmatched.remove(i)

                        foutput.write(FACE_TEMPLATE.format(
                            t=T, identifier=i, confidence=confidences[i],
                            left=int(face.left()), right=int(face.right()),
                            top=int(face.top()), bottom=int(face.bottom())))

                        faces[f] = None, None, None

            for _, face, _ in faces:

                # this face was matched already
                if face is None:
                    continue

                # new tracker
                tracker = dlib.correlation_tracker()
                tracker.start_track(gray, face)
                confidences[identifier] = tracker.update(gray)
                trackers[identifier] = tracker

                foutput.write(FACE_TEMPLATE.format(
                    t=T, identifier=identifier,
                    confidence=confidences[identifier],
                    left=int(face.left()), right=int(face.right()),
                    top=int(face.top()), bottom=int(face.bottom())))

                identifier = identifier + 1

            for i, tracker in trackers.items():

                if i not in unmatched:
                    continue

                face = tracker.get_position()

                foutput.write(FACE_TEMPLATE.format(
                    t=T, identifier=i, confidence=confidences[i],
                    left=int(face.left()), right=int(face.right()),
                    top=int(face.top()), bottom=int(face.bottom())))

def landmark(video, model, tracking, output, show_progress=False):
    """Facial features detection"""

    # frame generator
    frames = video.iterframes(with_time=True)
    if show_progress:
        frames = tqdm(iterable=frames,
                      total=video.duration * video.fps,
                      leave=True, mininterval=1.,
                      unit='frames', unit_scale=True)

    # face generator
    faceGenerator = getFaceGenerator(tracking)
    faceGenerator.send(None)

    # facial features detector
    facialFeaturesDetector = dlib.shape_predictor(model)

    with open(output, 'w') as foutput:

        for timestamp, frame in frames:

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # get all detected faces at this time
            T, faces = faceGenerator.send(timestamp)
            # not that T might be differ slightly from t
            # due to different steps in frame iteration

            for identifier, face, _ in faces:

                boundingBox = dlib.rectangle(
                    int(face.left()), int(face.top()),
                    int(face.right()), int(face.bottom()))
                points = facialFeaturesDetector(gray, boundingBox)
                facialFeatures = [(p.x, p.y) for p in points.parts()]

                foutput.write('{t:.3f} {identifier:d}'.format(
                    t=T, identifier=identifier))
                for x, y in facialFeatures:
                    foutput.write(' {x:d} {y:d}'.format(x=x, y=y))
                foutput.write('\n')

def features(video, model, shape, output, show_progress=False):
    """Openface FaceNet feature extraction"""

    face = Face(size=96, normalization='affine', openface=model)

    # frame generator
    frames = video.iterframes(with_time=True)
    if show_progress:
        frames = tqdm(iterable=frames,
                      total=video.duration * video.fps,
                      leave=True, mininterval=1.,
                      unit='frames', unit_scale=True)

    # shape generator
    shapeGenerator = getShapeGenerator(shape)
    shapeGenerator.send(None)

    with open(output, 'w') as foutput:

        for timestamp, frame in frames:

            T, shapes = shapeGenerator.send(timestamp)

            for identifier, landmarks in shapes:

                normalized = face._get_normalized(frame, landmarks)
                openface = face._get_openface(normalized)

                foutput.write('{t:.3f} {identifier:d}'.format(
                    t=T, identifier=identifier))
                for x in openface:
                    foutput.write(' {x:.5f}'.format(x=x))
                foutput.write('\n')


def get_fl(tracking):

    COLORS = [
        (240, 163, 255), (  0, 117, 220), (153,  63,   0), ( 76,   0,  92),
        ( 25,  25,  25), (  0,  92,  49), ( 43, 206,  72), (255, 204, 153),
        (128, 128, 128), (148, 255, 181), (143, 124,   0), (157, 204,   0),
        (194,   0, 136), (  0,  51, 128), (255, 164,   5), (255, 168, 187),
        ( 66, 102,   0), (255,   0,  16), ( 94, 241, 242), (  0, 153, 143),
        (224, 255, 102), (116,  10, 255), (153,   0,   0), (255, 255, 128),
        (255, 255,   0), (255,  80,   5)
    ]

    faceGenerator = getFaceGenerator(tracking)
    faceGenerator.send(None)

    def overlay(get_frame, timestamp):
        frame = get_frame(timestamp)
        height, width, _ = frame.shape
        _, faces = faceGenerator.send(timestamp)

        cv2.putText(frame, '{t:.3f}'.format(t=timestamp), (10, height-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1, 8, False)
        for identifier, face, confidence in faces:
            color = COLORS[identifier % len(COLORS)]

            # Draw face bounding box
            pt1 = (int(face.left()), int(face.top()))
            pt2 = (int(face.right()), int(face.bottom()))
            cv2.rectangle(frame, pt1, pt2, color, 2)

            # Print tracker identifier
            cv2.putText(frame, '#{identifier:d}'.format(identifier=identifier),
                        (pt1[0] + 2, pt1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, 8, False)

            # Print tracker confidence
            cv2.putText(frame,
                        '{confidence:d}'.format(confidence=int(confidence)),
                        (pt1[0] + 2, pt2[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, 8, False)
        return frame

    return overlay


def demo(filename, tracking, output, t_start=0, t_end=None):

    import os
    os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'
    from moviepy.video.io.VideoFileClip import VideoFileClip

    original_clip = VideoFileClip(filename)
    modified_clip = original_clip.fl(get_fl(tracking))
    cropped_clip = modified_clip.subclip(t_start=t_start, t_end=t_end)
    cropped_clip.write_videofile(output, audio=False)


if __name__ == '__main__':

    # parse command line arguments
    version = 'pyannote-face {version}'.format(version=__version__)
    arguments = docopt(__doc__, version=version)

    # initialize video
    filename = arguments['<video>']
    video = Video(filename)

    verbose = arguments['--verbose']

    # face detection
    if arguments['detect']:

        # every xxx milliseconds
        every = arguments['--every']
        if not every:
            step = None
        else:
            step = 1e-3 * float(arguments['--every'])

        # (approximate) size of smallest face
        smallest = float(arguments['--smallest'])
        if smallest > SMALLEST_DEFAULT:
            upscale = 1
        else:
            upscale = int(np.ceil(SMALLEST_DEFAULT / smallest))

        output = arguments['<output>']

        detect(video, output,
               step=step, upscale=upscale,
               show_progress=verbose)

    # face tracking
    if arguments['track']:

        shot = arguments['<shot>']
        detection = arguments['<detection>']
        output = arguments['<output>']
        min_overlap_ratio = float(arguments['--min-overlap'])
        min_confidence = float(arguments['--min-confidence'])
        track(video, shot, detection, output,
              min_overlap_ratio=min_overlap_ratio,
              min_confidence=min_confidence, show_progress=verbose)

    # facial features detection
    if arguments['landmark']:

        tracking = arguments['<tracking>']
        model = arguments['<model>']
        output = arguments['<output>']
        landmark(video, model, tracking, output,
              show_progress=verbose)

    if arguments['features']:

        model = arguments['<model>']
        shape = arguments['<landmark>']
        output = arguments['<output>']
        features(video, model, shape, output,
                 show_progress=verbose)

    if arguments['demo']:

        tracking = arguments['<tracking>']
        output = arguments['<output>']

        t_start = float(arguments['--from'])
        t_end = arguments['--until']
        t_end = float(t_end) if t_end else None

        demo(filename, tracking, output, t_start=t_start, t_end=t_end)
