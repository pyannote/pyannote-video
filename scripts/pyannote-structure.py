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


"""Video structure

The standard pipeline for is the following:

    shot boundary detection ==> shot threading ==> segmentation into scenes

Usage:
  pyannote-structure.py shot [options] <video> <output.json>
  pyannote-structure.py thread [options] <video> <shot.json> <output.json>
  pyannote-structure.py scene [options] <video> <thread.json> <output.json>
  pyannote-structure.py (-h | --help)
  pyannote-structure.py --version

Options:
  --height=<n_pixels>    Resize video frame to height <n_pixels> [default: 50].
  --window=<n_seconds>   Apply median filtering on <n_seconds> window [default: 2.0].
  --threshold=<value>    Set threshold to <value> [default: 1.0].
  --min-match=<n_match>  Set minimum number of matches to <n_match> [default: 20].
  --lookahead=<n_shots>  Look at up to <n_shots> following shots [default: 24].
  -h --help              Show this screen.
  --version              Show version.
  --verbose              Show progress.
"""

from docopt import docopt

from pyannote.core import Timeline
import pyannote.core.json

from pyannote.video import __version__
from pyannote.video import Video
from pyannote.video import Shot, Thread


def do_shot(video, output, height=50, window=2.0, threshold=1.0):

    shots = Shot(video, height=height, context=window, threshold=threshold)
    shots = Timeline(shots)
    with open(output, 'w') as fp:
        pyannote.core.json.dump(shots, fp)

def do_thread(video, shots, output, min_match=20, lookahead=24, verbose=False):

    with open(shots, 'r') as fp:
        shots = pyannote.core.json.load(fp)
    threads = Thread(video, shot=shots, lookahead=lookahead,
                     min_match=min_match, verbose=verbose)
    threads = threads()
    with open(output, 'w') as fp:
        pyannote.core.json.dump(threads, fp)

def do_scene(video, threads, output, verbose=False):

    with open(threads, 'r') as fp:
        threads = pyannote.core.json.load(fp)
    raise NotImplementedError('Not yet available')
    # scenes = Scene(video, thread=threads, verbose=verbose)
    # with open(output, 'w') as fp:
    #     pyannote.core.json.dump(scenes, fp)


if __name__ == '__main__':

    # parse command line arguments
    version = 'pyannote-structure {version}'.format(version=__version__)
    arguments = docopt(__doc__, version=version)

    # common 'verbosity' option
    verbose = arguments['--verbose']

    # common 'output' argument
    output = arguments['<output.json>']

    # initialize video
    filename = arguments['<video>']
    video = Video(filename, verbose=verbose)

    if arguments['shot']:
        height = int(arguments['--height'])
        window = float(arguments['--window'])
        threshold = float(arguments['--threshold'])
        do_shot(video, output,
                height=height, window=window, threshold=threshold)

    if arguments['thread']:
        shots = arguments['<shot.json>']
        min_match = int(arguments['--min-match'])
        lookahead = int(arguments['--lookahead'])
        do_thread(video, shots, output,
                  min_match=min_match, lookahead=lookahead, verbose=verbose)

    if arguments['scene']:
        threads = arguments['<thread.json>']
        do_scene(video, threads, output, verbose=verbose)
