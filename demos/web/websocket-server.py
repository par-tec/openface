#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
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

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.python import log
from twisted.internet import reactor

import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import os
import StringIO
import urllib
import base64
import pickle
import time
from struct import pack

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import openface

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

args = parser.parse_args()

# Recognize the face using the dlibFacePredictor model.
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)


class Face:
    """A container class with debug code."""
    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


def _stream_to_rgbFrame(imgdata):
    """Return an rgbframe from a byte-sequence.
    """
    imgF = StringIO.StringIO()
    imgF.write(imgdata)
    imgF.seek(0)
    # import pdb; pdb.set_trace()
    img = Image.open(imgF)
    buf = np.fliplr(np.asarray(img))
    rgbFrame = np.zeros((img.size[1], img.size[0], 3), dtype=np.uint8)
    rgbFrame[:, :, 0] = buf[:, :, 2]
    rgbFrame[:, :, 1] = buf[:, :, 1]
    rgbFrame[:, :, 2] = buf[:, :, 0]
    return rgbFrame


def getRep(imgPath):
    """Detect a face in an image and returns the aligned representation."""

    try:
        is_file = os.path.isfile(imgPath)
    except (TypeError, UnicodeError, OSError, IOError) as e:
        is_file = False

    if is_file:
        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    else:
        rgbImg = _stream_to_rgbFrame(imgPath)
        imgPath = "<datastream>"

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))

    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))

    rep = net.forward(alignedFace)
    return rep

from collections import defaultdict

class OpenFaceServerProtocol(WebSocketServerProtocol):

    def __init__(self):
        self.images = {}
        self.training = True
        self.people = []

        # Use a zero face.
        self.cid = np.zeros(128)
        self.cid_stats = self._reset_stats()

        if args.unknown:
            self.unknownImgs = np.load("./examples/web/unknown.npy")

    def _reset_stats(self):
        return {
            'max': 0.0,
            'min': 2.0,
            'history': []
        }

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = True

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        try:
            raw = payload.decode('utf8')
            msg = json.loads(raw)
        except Exception as e:
            print("Error processing msg: %r" % payload)
            print(e)
            return

        print("Received {} message of length {}.".format(
            msg['type'], len(raw)))
        if msg['type'] == "ALL_STATE":
            self.loadState(msg['images'], msg['training'], msg['people'])
        elif msg['type'] == "NULL":
            self.sendMessage('{"type": "NULL"}')
        elif msg['type'] == "FRAME":
            self.processFrame(msg['dataURL'], msg['identity'])
            self.sendMessage('{"type": "PROCESSED"}')
        elif msg['type'] == "TRAINING":
            self.training = msg['val']
            if not self.training:
                self.trainSVM()
        elif msg['type'] == "SET_IDCARD":
            payload = b''.join([pack(">B", x) for x in map(ord, msg['val'])])
            self.cid = getRep(payload)
            self.cid_stats = self._reset_stats()
        else:
            print("Warning: Unknown message type: {!r}".format(payload))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def loadState(self, jsImages, training, jsPeople):
        self.training = training

        for jsImage in jsImages:
            h = jsImage['hash'].encode('ascii', 'ignore')
            self.images[h] = Face(np.array(jsImage['representation']),
                                  jsImage['identity'])

        for jsPerson in jsPeople:
            self.people.append(jsPerson.encode('ascii', 'ignore'))

    @staticmethod
    def _figure_to_stream():
        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='png')
        plt.close()
        imgdata.seek(0)
        content = 'data:image/png;base64,' + \
                  urllib.quote(base64.b64encode(imgdata.buf))
        return content

    def processFrame(self, dataURL, identity):
        """Extract an image from the data passed to the server via js:
            - in training mode, analyze and archive
            - otherwise try to predict the identity.
        """
        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)

        buf = np.fliplr(np.asarray(img))
        rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]

        # Prepare the image to be presented on the screen.
        annotatedFrame = np.copy(buf)

        bb = align.getLargestFaceBoundingBox(rgbFrame)

        if bb is None:
            return

        landmarks = align.findLandmarks(rgbFrame, bb)
        alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                  landmarks=landmarks,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            return

        # Create a perceptive hash from the webcam and look
        #  into the actual images. It seems rarely used.
        phash = str(imagehash.phash(Image.fromarray(alignedFace)))
        if phash in self.images:
            rep = self.images[phash].rep
        else:
            print("Phash not found in cache. Calculate the face representation.")
            rep = net.forward(alignedFace)

        # Create the annotated frame.
        bl = (bb.left(), bb.bottom())
        tr = (bb.right(), bb.top())
        cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                      thickness=3)
        for p in openface.AlignDlib.OUTER_EYES_AND_NOSE:
            cv2.circle(annotatedFrame, center=landmarks[p], radius=3,
                       color=(102, 204, 255), thickness=-1)

        # Distance^2 between the face in the idcard and the webcam.
        d = self.cid - rep
        d2 = np.dot(d, d)
        self.cid_stats = {
            'min': min(self.cid_stats['min'], d2),
            'max': max(self.cid_stats['max'], d2)
        }
        name = "d: {:0.2f}\n" \
               "M: {:0.2f}\n" \
               "m: {:0.2f}".format(np.sqrt(d2), self.cid_stats['max'], self.cid_stats['min'])
        cv2.putText(annotatedFrame, name, (bb.left(), bb.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                    color=(152, 255, 204), thickness=2)

        content = self._create_figure_from_frame(annotatedFrame)
        msg = {
            "type": "ANNOTATED",
            "content": content
        }
        self.sendMessage(json.dumps(msg))

    @staticmethod
    def _create_figure_from_frame(annotatedFrame):
        plt.figure()
        plt.imshow(annotatedFrame)
        plt.xticks([])
        plt.yticks([])
        content = OpenFaceServerProtocol._figure_to_stream()
        return content


if __name__ == '__main__':
    log.startLogging(sys.stdout)

    factory = WebSocketServerFactory("ws://localhost:{}".format(args.port),
                                     debug=False)
    factory.protocol = OpenFaceServerProtocol

    reactor.listenTCP(args.port, factory)
    reactor.run()
