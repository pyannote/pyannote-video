# Most code in this file was shamelessly borrowed from MoviePy
# http://zulko.github.io/moviepy/

# The MIT License (MIT)
#
# Copyright (c) 2015 Zulko
# Copyright (c) 2015 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
This module implements all the functions to read a video or a picture
using ffmpeg. It is quite ugly, as there are many pitfalls to avoid
"""

from __future__ import division

import subprocess as sp
import os
import re
import warnings
import logging
import numpy as np

logging.captureWarnings(True)

try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


def _is_string(obj):
    """ Returns true if s is string or string-like object,
    compatible with Python 2 and Python 3."""
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)


def _cvsecs(time):
    """ Will convert any time into seconds.
    Here are the accepted formats:
    >>> _cvsecs(15.4) -> 15.4 # seconds
    >>> _cvsecs( (1,21.5) ) -> 81.5 # (min,sec)
    >>> _cvsecs( (1,1,2) ) -> 3662 # (hr, min, sec)
    >>> _cvsecs('01:01:33.5') -> 3693.5  #(hr,min,sec)
    >>> _cvsecs('01:01:33.045') -> 3693.045
    >>> _cvsecs('01:01:33,5') #coma works too
    """

    if _is_string(time):
        if (',' not in time) and ('.' not in time):
            time = time + '.0'
        expr = r"(\d+):(\d+):(\d+)[,|.](\d+)"
        finds = re.findall(expr, time)[0]
        nums = list(map(float, finds))
        return (3600*int(finds[0]) +
                60*int(finds[1]) +
                int(finds[2]) +
                nums[3]/(10**len(finds[3])))

    elif isinstance(time, tuple):
        if len(time) == 3:
            hr, mn, sec = time
        elif len(time) == 2:
            hr, mn, sec = 0, time[0], time[1]
        return 3600*hr + 60*mn + sec

    else:
        return time


class Video:

    def __init__(self, filename, ffmpeg='ffmpeg', debug=False):

        self.filename = filename
        self.ffmpeg = ffmpeg
        infos = self._parse_infos(print_infos=debug, check_duration=True)
        self.fps = infos['video_fps']
        self.size = infos['video_size']
        self.duration = infos['video_duration']
        self.ffmpeg_duration = infos['duration']
        self.nframes = infos['video_nframes']

        self.infos = infos

        self.pix_fmt = 'rbg24'
        self.depth = 3

        w, h = self.size
        bufsize = self.depth * w * h + 100

        self.bufsize = bufsize
        self._initialize()

        self.pos = 1
        self.lastread = self._read_frame()

    def _parse_infos(self, print_infos=False, check_duration=True):
        """Get file infos using ffmpeg.

        Returns a dictionnary with the fields:
        "video_found", "video_fps", "duration", "video_nframes",
        "video_duration", "audio_found", "audio_fps"

        "video_duration" is slightly smaller than "duration" to avoid
        fetching the uncomplete frames at the end, which raises an error.

        """

        # open the file in a pipe, provoke an error, read output
        is_GIF = self.filename.endswith('.gif')
        cmd = [self.ffmpeg, "-i", self.filename]
        if is_GIF:
            cmd += ["-f", "null", "/dev/null"]

        popen_params = {"bufsize": 10**5,
                        "stdout": sp.PIPE,
                        "stderr": sp.PIPE,
                        "stdin": DEVNULL}

        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000

        proc = sp.Popen(cmd, **popen_params)

        proc.stdout.readline()
        proc.terminate()
        infos = proc.stderr.read().decode('utf8')
        del proc

        if print_infos:
            # print the whole info text returned by FFMPEG
            print(infos)

        lines = infos.splitlines()
        if "No such file or directory" in lines[-1]:
            raise IOError(
                ("MoviePy error: the file %s could not be found !\n"
                 "Please check that you entered the correct path.") % self.filename)

        result = dict()

        # get duration (in seconds)
        result['duration'] = None

        if check_duration:
            try:
                keyword = ('frame=' if is_GIF else 'Duration: ')
                line = [l for l in lines if keyword in l][0]
                match = re.findall(
                    "([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])", line)[0]
                result['duration'] = _cvsecs(match)
            except:
                raise IOError(
                    ("MoviePy error: failed to read the duration of file %s.\n"
                     "Here are the file infos returned by ffmpeg:\n\n%s") % (
                                  self.filename, infos))

        # get the output line that speaks about video
        lines_video = [l for l in lines
                       if ' Video: ' in l and re.search('\d+x\d+', l)]

        result['video_found'] = lines_video != []

        if result['video_found']:

            line = lines_video[0]

            try:

                # get the size, of the form 352x288 (w x h)
                m = re.search(" [0-9]*x[0-9]*(,| )", line)
                s = [int(p) for p in line[m.start():m.end()-1].split('x')]
                result['video_size'] = s

            except:
                message = (
                    "MoviePy error: failed to read video dimensions in file "
                    "%s.\nHere are the file infos returned by ffmpeg:\n\n%s"
                )
                raise IOError(message % (self.filename, infos))

            try:

                # get the display aspect ratio, of the form 178:163
                m = re.search("DAR [0-9]*:[0-9]*", line)
                s = [int(p) for p in line[m.start()+4:m.end()].split(':')]
                result['video_dar'] = s

            except:
                result['video_dar'] = None

            # get the frame rate. Sometimes it's 'tbr', sometimes 'fps',
            # sometimes tbc, and sometimes tbc/2...
            # Current policy: Trust tbr first, then fps. If result is near
            # from x*1000/1001 where x is 23,24,25,50, replace by x*1000/1001
            # (very common case for the fps).

            try:
                match = re.search("( [0-9]*.| )[0-9]* tbr", line)
                tbr = float(line[match.start():match.end()].split(' ')[1])
                result['video_fps'] = tbr

            except:
                match = re.search("( [0-9]*.| )[0-9]* fps", line)
                result['video_fps'] = float(line[match.start():match.end()].split(' ')[1])

            # It is known that a fps of 24 is often written as 24000/1001
            # but then ffmpeg nicely rounds it to 23.98, which we hate.
            coef = 1000.0/1001.0
            fps = result['video_fps']
            for x in [23, 24, 25, 30, 50]:
                if (fps!=x) and abs(fps - x*coef) < .01:
                    result['video_fps'] = x*coef

            if check_duration:
                result['video_nframes'] = int(result['duration']*result['video_fps'])+1
                result['video_duration'] = result['duration']
            else:
                result['video_nframes'] = 1
                result['video_duration'] = None

            # We could have also recomputed the duration from the number
            # of frames, as follows:
            # >>> result['video_duration'] = result['video_nframes'] / result['video_fps']

        lines_audio = [l for l in lines if ' Audio: ' in l]

        result['audio_found'] = lines_audio != []

        if result['audio_found']:
            line = lines_audio[0]
            try:
                match = re.search(" [0-9]* Hz", line)
                result['audio_fps'] = int(line[match.start()+1:match.end()])
            except:
                result['audio_fps'] = 'unknown'

        return result

    def _initialize(self, t=0):
        """Opens the file, creates the pipe. """

        self._close()  # if any

        if t != 0:
            offset = min(1, t)
            i_arg = ['-ss', "%.06f" % (t - offset),
                     '-i', self.filename,
                     '-ss', "%.06f" % offset]
        else:
            i_arg = ['-i', self.filename]

        cmd = (
            [self.ffmpeg] + i_arg +
            ['-loglevel', 'error', '-f', 'image2pipe',
             '-pix_fmt', self.pix_fmt, '-vcodec', 'rawvideo', '-'])

        popen_params = {"bufsize": self.bufsize,
                        "stdout": sp.PIPE,
                        "stderr": sp.PIPE,
                        "stdin": DEVNULL}

        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000

        self.proc = sp.Popen(cmd, **popen_params)

    def _skip_frames(self, n=1):
        """Reads and throws away n frames """
        w, h = self.size
        for _ in range(n):
            self.proc.stdout.read(self.depth*w*h)
            # self.proc.stdout.flush()
        self.pos += n

    def _read_frame(self):
        w, h = self.size
        nbytes = self.depth*w*h

        s = self.proc.stdout.read(nbytes)

        if len(s) != nbytes:

            warnings.warn(
                "Warning: in file %s, " % (self.filename) +
                "%d bytes wanted but %d bytes read," % (nbytes, len(s)) +
                "at frame %d/%d, at time %.02f/%.02f sec. " % (
                    self.pos, self.nframes, 1.0 * self.pos / self.fps,
                    self.duration) +
                "Using the last valid frame instead.",
                UserWarning)

            if not hasattr(self, 'lastread'):
                message = (
                    "MoviePy error: failed to read the first frame of "
                    "video file %s. That might mean that the file is "
                    "corrupted. That may also mean that you are using "
                    "a deprecated version of FFMPEG. On Ubuntu/Debian "
                    "for instance the version in the repos is deprecated. "
                    "Please update to a recent version from the website."
                )
                raise IOError(message % self.filename)

            result = self.lastread

        else:

            result = np.fromstring(s, dtype='uint8')
            # reshape((h, w, len(s)//(w*h)))
            result.shape = (h, w, len(s)//(w*h))
            self.lastread = result

        return result

    def iterframes(self, start=None, end=None, step=None, with_time=False):

        # default: starts from the beginning
        if start is None:
            start = 0.

        # default: iterates until the end
        if end is None:
            end = self.duration

        # default: frame by frame
        if step is None:
            step = 1./self.fps

        # TODO warning if step != N x 1/fps (where N is an integer)
        # warnings.warn(message, UserWarning)

        for t in np.arange(start, end, step):
            frame = self._get_frame(t)

            if with_time:
                yield t, frame
            else:
                yield frame

    def __call__(self, t):
        return self._get_frame(t)

    def _get_frame(self, t):
        """ Read a file video frame at time t.

        Note for coders: getting an arbitrary frame in the video with
        ffmpeg can be painfully slow if some decoding has to be done.
        This function tries to avoid fectching arbitrary frames
        whenever possible, by moving between adjacent frames.
        """

        # these definitely need to be rechecked sometime. Seems to work.

        # I use that horrible '+0.00001' hack because sometimes due to
        # numerical imprecisions a 3.0 can become a 2.99999999... which makes
        # the int() go to the previous integer. This makes the fetching more
        # robust in the case where you get the nth frame by writing
        # _get_frame(n/fps).

        pos = int(self.fps*t + 0.00001)+1

        if pos == self.pos:
            return self.lastread
        else:
            if(pos < self.pos) or (pos > self.pos+100):
                self._initialize(t)
                self.pos = pos
            else:
                self._skip_frames(pos-self.pos-1)
            result = self._read_frame()
            self.pos = pos
            return result

    def _close(self):
        if hasattr(self, 'proc'):
            self.proc.terminate()
            self.proc.stdout.close()
            self.proc.stderr.close()
            del self.proc

    def __del__(self):
        self._close()
        if hasattr(self, 'lastread'):
            del self.lastread
