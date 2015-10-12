"""Shot boundary detection

Usage:
  pyannote-shot.py [options] <video> <shot>
  pyannote-shot.py (-h | --help)
  pyannote-shot.py --version

Options:
  -h --help            Show this screen.
  --version            Show version.
  --verbose            Show progress.
  --pre=<path>         Load/save preprocessed displaced frame difference.
  --threshold=<float>  Set peak detection threshold [default: 1.2]
"""
from docopt import docopt

from pyannote.video import __version__
from pyannote.video import Video

import cv2
import numpy as np
import itertools
import os.path

import scipy.signal

from tqdm import tqdm


def _dfd(previous, current, flow=None):
    """Displaced frame difference"""

    flow = cv2.calcOpticalFlowFarneback(
        previous, current, flow, 0.5, 3, 15, 3, 5, 1.1, 0)

    height, width = previous.shape
    reconstruct = np.zeros(previous.shape)

    for x, y in itertools.product(range(width), range(height)):
        dy, dx = flow[y, x]
        rx = max(0, min(x + dx, width - 1))
        ry = max(0, min(y + dy, height - 1))
        reconstruct[y, x] = current[ry, rx]

    return np.mean(np.abs(previous - reconstruct))


def convert(rgb, n=8):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    return cv2.resize(gray, (height / n, width / n))


def preprocess(video, show_progress=False):
    """Shot boundary detection based on displaced frame difference"""

    previous = None
    T = []
    D = []

    # frame iterator
    generator = video.iterframes(with_time=True)
    if show_progress:
        generator = tqdm(iterable=generator,
                         total=video.duration * video.fps,
                         leave=True, mininterval=1.,
                         unit='frames', unit_scale=True)

    # iterate frames one by one
    for t, frame in generator:

        current = convert(frame)

        if previous is None:
            previous = current
            continue

        T.append(t)
        D.append(_dfd(previous, current, flow=None))

        previous = current

    return T, D


def process(timestamps, differences, threshold):

    # temporal smoothing
    filtered = scipy.signal.medfilt(differences, kernel_size=49)

    # normalized displaced frame difference
    normalized = (differences - filtered) / filtered

    # apply threshold on normalized displaced frame difference
    # in case multiple consecutive value are higher than the threshold,
    # only keep the first one as a shot boundary.
    boundaries = []
    _i = 0
    for i in np.where(normalized > threshold)[0]:
        if i == _i + 1:
            _i = i
            continue
        boundaries.append(timestamps[i])
        _i = i

    return boundaries


if __name__ == '__main__':

    # parse command line arguments
    version = 'pyannote-shot {version}'.format(version=__version__)
    arguments = docopt(__doc__, version=version)
    filename = arguments['<video>']
    shot = arguments['<shot>']
    verbose = arguments['--verbose']

    loadFrom = saveTo = arguments['--pre']
    threshold = float(arguments['--threshold'])

    if loadFrom and os.path.exists(loadFrom):

        timestamps = []
        differences = []
        with open(loadFrom, 'r') as f:
            for line in f:
                t, d = line.strip().split()
                timestamps.append(float(t))
                differences.append(float(d))

    else:

        # initialize video
        video = Video(filename)

        # compute displaced frame difference
        timestamps, differences = preprocess(video, show_progress=verbose)

        # save displaced frame difference
        if saveTo:
            with open(saveTo, 'w') as f:
                for t, d in itertools.izip(timestamps, differences):
                    f.write("{t:.3f} {d:.3f}\n".format(t=t, d=d))

    # peak detection
    boundaries = process(timestamps, differences, threshold)

    # save results to file
    with open(shot, 'w') as f:
        for boundary in boundaries:
            f.write('{boundary:.3f}\n'.format(boundary=boundary))
