# Copyright 2015 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import dlib
import cv2
from .openface import TorchWrap


SMALLEST_DEFAULT = 36

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])


EYES_AND_BOTTOM_LIP = np.array([39, 42, 57])


class Face(object):

    def __init__(self, landmarks=None, smallest=SMALLEST_DEFAULT, size=96,
                 openface=None):

        # face detection
        self._faceDetector = dlib.get_frontal_face_detector()
        if smallest > SMALLEST_DEFAULT:
            self._upscale = 1
        else:
            self._upscale = int(np.ceil(SMALLEST_DEFAULT / smallest))

        self._landmarksDetector = dlib.shape_predictor(landmarks)

        self.size = size
        self._landmarks = TEMPLATE

        if openface is not None:
            self._net = TorchWrap(model=openface, size=self.size, cuda=False)

    def iterfaces(self, bgr):
        """Iterate over all detected faces"""
        for face in self._faceDetector(bgr, self._upscale):
            yield face

    def getLargestFaceBoundingBox(self, bgr):
        faces = list(self.iterfaces(bgr))
        if len(faces) > 0:
            return max(faces, key=lambda rect: rect.width() * rect.height())

    def _get_landmarks(self, bgr, face):
        points = self._landmarksDetector(bgr, face)
        return list(map(lambda p: (p.x, p.y), points.parts()))

    def normalize(self, bgr, face):

        alignPoints = self._get_landmarks(bgr, face)
        meanAlignPoints = self.transformPoints(self._landmarks, face, True)

        (xs, ys) = zip(*meanAlignPoints)
        tightBb = dlib.rectangle(left=min(xs), right=max(xs),
                                 top=min(ys), bottom=max(ys))

        npAlignPoints = np.float32(alignPoints)
        npMeanAlignPoints = np.float32(meanAlignPoints)

        npAlignPointsSS = npAlignPoints[EYES_AND_BOTTOM_LIP]
        npMeanAlignPointsSS = npMeanAlignPoints[EYES_AND_BOTTOM_LIP]
        H = cv2.getAffineTransform(npAlignPointsSS, npMeanAlignPointsSS)
        warpedImg = cv2.warpAffine(bgr, H, np.shape(bgr)[0:2])

        wBb = self.getLargestFaceBoundingBox(warpedImg)
        if wBb is None:
            return
        wAlignPoints = self._get_landmarks(warpedImg, wBb)
        wMeanAlignPoints = self.transformPoints(
            self._landmarks, wBb, True)

        if len(warpedImg.shape) != 3:
            print("  + Warning: Result does not have 3 dimensions.")
            return None

        (xs, ys) = zip(*wAlignPoints)
        xRange = max(xs) - min(xs)
        yRange = max(ys) - min(ys)
        (l, r, t, b) = (min(xs), max(xs), min(ys), max(ys))
        (w, h, _) = warpedImg.shape
        if 0 <= l <= w and 0 <= r <= w and 0 <= b <= h and 0 <= t <= h:
            cwImg = cv2.resize(warpedImg[t:b, l:r], (self.size, self.size))
            h, edges = np.histogram(cwImg.ravel(), 16, [0, 256])
            s = sum(h)
            if any(h > 0.65 * s):
                print("Warning: Image is likely a single color.")
                return
        else:
            print("Warning: Unable to align and crop to the "
                  "face's bounding box.")
            return

        return cwImg

    def _get_openface(self, normalized):
        return self._net.forwardImage(normalized)

    def openface(self, bgr, face):
        normalized = self.normalize(bgr, face)
        if normalized is None:
            return [0.] * 128
        return self._get_openface(normalized)

    @staticmethod
    def transformPoints(points, face, toImgCoords):
        if toImgCoords:
            def scale(p):
                (x, y) = p
                return (int((x * face.width()) + face.left()),
                        int((y * face.height()) + face.top()))
        else:
            def scale(p):
                (x, y) = p
                return (float(x - face.left()) / face.width(),
                        float(y - face.top()) / face.height())
        return list(map(scale, points))
