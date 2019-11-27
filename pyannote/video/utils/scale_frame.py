#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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
# Paul LERNER

import dlib

def scale_up_bbox(bbox,frame_width,frame_height):
    left, top, right, bottom=bbox
    left = int(left * frame_width)
    right = int(right * frame_width)
    top = int(top * frame_height)
    bottom = int(bottom * frame_height)
    return left, top, right, bottom

def bbox_to_rectangle(bbox,frame_width,frame_height, double=True):
    rectangle = dlib.drectangle if double else dlib.rectangle
    left, top, right, bottom=scale_up_bbox(bbox,frame_width,frame_height)
    face = rectangle(left, top, right, bottom)
    return face

def rectangle_to_bbox(rectangle,frame_width,frame_height):
    left, top, right, bottom=rectangle.left(),rectangle.top(),rectangle.right(),rectangle.bottom()
    left/=frame_width
    top/=frame_height
    right/=frame_width
    bottom/=frame_height
    return (left, top, right, bottom)

def parts_to_landmarks(landmarks,frame_width,frame_height):
    save_landmarks=[]
    for p in landmarks.parts():
        x, y = p.x, p.y
        save_landmarks.append((x / frame_width, y / frame_height))
    return save_landmarks
