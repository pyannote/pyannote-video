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

Usage:
  pyannote-face detect [options] <video> <output>
  pyannote-face track [options] <video> <shot> <face> <track>
  pyannote-face (-h | --help)
  pyannote-face --version

Options:
  --every=<msec>       Process one frame every <msec> milliseconds.
  --shape=<model>      Perform facial features detection using <model>.
  -h --help            Show this screen.
  --version            Show version.
  --verbose            Show progress.
"""

from docopt import docopt
from pyannote.video import __version__
from pyannote.video import Video

from tqdm import tqdm
import cv2

import dlib


def detect(video, output, step=None, shape=None, show_progress=False):

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

    # facial features detector
    if shape is None:
        facialFeaturesDetector = None
    else:
        facialFeaturesDetector = dlib.shape_predictor(shape)

    with open(output, 'w') as f:

        for t, frame in generator:

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            for boundingBox in faceDetector(gray, 1):

                f.write('{t:.3f} {left:d} {top:d} {right:d} {bottom:d}'.format(
                    t=t, left=boundingBox.left(), right=boundingBox.right(),
                    top=boundingBox.top(), bottom=boundingBox.bottom()))

                if facialFeaturesDetector:

                    points = facialFeaturesDetector(gray, boundingBox)
                    facialFeatures = [(p.x, p.y) for p in points.parts()]

                    for x, y in facialFeatures:
                        f.write(' {x:d} {y:d}'.format(x=x, y=y))

                f.write('\n')
                f.flush()

def track(video):
    pass


if __name__ == '__main__':

    # parse command line arguments
    version = 'pyannote-face {version}'.format(version=__version__)
    arguments = docopt(__doc__, version=version)

    # initialize video
    filename = arguments['<video>']
    video = Video(filename)

    output = arguments['<output>']
    verbose = arguments['--verbose']

    every = arguments['--every']
    if not every:
        step = None
    else:
        step = 1e-3 * float(arguments['--every'])

    # facial features detection
    shape = arguments['--shape']
    if not shape:
        shape = None

    if arguments['detect']:
        detect(video, output,
               step=step, shape=shape,
               show_progress=verbose)

    if arguments['track']:
        track(video, show_progress=verbose)
