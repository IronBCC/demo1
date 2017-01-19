


import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)


def import_module(path):
    sys.path.insert(0, path)

import_module("deepcut-cnn/python")
import_module("deepcut-cnn/python/pose")
import_module("mxnet_ssd")
import_module("openface")
import_module("openface/demos")

import argparse

import cv2
import time
import glob
import numpy as np

from tools import find_mxnet
from detect.detector import Detector

from estimate_pose import estimate_pose

import mxnet as mx
import caffe

import classifier_webcam
import openface

import os
import importlib
import sys

import json

import shutil
import os

import math

import scipy

import activity_detection

from collections import namedtuple

from mxnet.io import DataBatch

import rx
from rx import Observable
from rx.subjects import Subject
from rx.core import Scheduler

def print_it(x):
    print x



CACHE_FOLDER = "cache/"

Batch = namedtuple('Batch', ['data'])

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
# CLASSES = (None, None, None, None,
#            None, None, None, 'cat', 'chair',
#            None, None, 'dog', None,
#            None, 'person', None,
#            None, 'sofa', None, None)

MAX_IMAGE_SIZE = 700
SSD_SHAPE_SIZE = 300
SKIP_RATE = 12
CLEAR_RATE = SKIP_RATE * 60


def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx,
                 nms_thresh=0.5, force_nms=True):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    force_nms : bool
        force suppress different categories
    """
    import_module("mxnet_ssd/symbol")
    net = importlib.import_module("symbol_" + net) \
        .get_symbol(len(CLASSES), nms_thresh, force_nms)
    detector = Detector(net, prefix + "_" + str(data_shape), epoch, \
        data_shape, mean_pixels, ctx=ctx)
    return detector

def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection network demo')
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        choices=['vgg16_reduced'], help='which network to use')
    parser.add_argument('--images', dest='images', type=str, default='./data/demo/dog.jpg',
                        help='run demo with images, use comma(without space) to seperate multiple images')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='demo image directory, optional', type=str)
    parser.add_argument('--ext', dest='extension', help='image extension, optional',
                        type=str, nargs='?')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'mxnet_ssd/model', 'ssd'), type=str)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=SSD_SHAPE_SIZE,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.5,
                        help='object visualize score threshold, default 0.6')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--timer', dest='show_timer', type=bool, default=True,
                        help='show detection time')
    parser.add_argument('--visualize', dest='visualize', type=bool, default=True,
                        help='show detection time')
    parser.add_argument('--verbose', dest='verbose', type=bool, default=False,
                        help='show detection time')
    parser.add_argument('--classifierModel', dest='classifierModel', type=str, default="openface/models/openface/celeb-classifier.nn4.small2.v1.pkl",
                        help='show detection time')
    parser.add_argument('--imgDim', dest='imgDim', type=int, default=96,
                        help='show detection time')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    return args



def extract_rect(det, width, height, tresh = 0.6, classes = CLASSES):
    class_id = int(det[0])
    if det[1] < tresh or (not class_id and class_id >= len(classes)):
        return None, None, None
    class_id = classes[class_id]
    xmin = int(det[2] * width)
    ymin = int(det[3] * height)
    xmax = int(det[4] * width)
    ymax = int(det[5] * height)
    return (xmin, ymin), (xmax, ymax), class_id



def predict(filename, mod, synsets):
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)

# {
#   "persons":{
#     [
#       {
#         "id": 102039,
#         "poseCoordinats":{},
#         "pose": "standing, hand up",
#         "emotions": "nervious",
#         "secondsSinceAppearing": 2100
#       },
#       {
#         "id": 10439,
#         "poseCoordinats":{},
#         "pose": "laying",
#         "emotions": "neutral",
#         "secondsSinceAppearing": 2100
#       },
#     ]
#   }
# }

###########
# Pose
###########
# 0 - right foot
# 1 - right knee
# 2 - right hip
# 3 - left hip
# 4 - left knee
# 5 - left foot
# 6 - right palm
# 7 - right elbow
# 8 - right shoulder
# 9 - left shoulder
# 10 - left elbow
# 11 - left palm
# 12 - chin
# 13 - forehead
##############

def _npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    image[cy-radius:cy+radius, cx-radius:cx+radius][index] = (
        image[cy-radius:cy+radius, cx-radius:cx+radius][index].astype('float32') * transparency +
        np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')

def resize(img, size):
    size = (min(size[0], img.shape[0]), min(size[1], img.shape[1]))
    r = img.shape[0] / float(img.shape[1])
    size = (size[0], int(size[0] * r))
    return cv2.resize(img, size), r

def face_rect(img, pose):
    face = ((pose[0, 13], pose[1, 13]), (pose[0, 12], pose[1, 12]))
    dt = face[1][1] - face[0][1]
    return ((int(face[0][1]), int(face[0][0] - dt)), (int(face[1][1] + dt), int(face[1][0] + dt)))
    # return img[face[0][0] - dt:face[1][0] + dt, face[0][1]:face[1][1] + dt]
    # return resize(img, (SSD_SHAPE_SIZE, SSD_SHAPE_SIZE))

def pose_to_global(pose, person):
    return pose

def eval_face(img, face_bb):
    start = time.clock()
    alignedFace = align.align(args.imgDim, img, face_bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    if alignedFace is None:
        raise Exception("Unable to align the frame")
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    return net.forward(alignedFace)

FACES_DB = []
FACE_TRESHOLD = 0.8

FLIP = False

def clearCacheFolder():
    print "Clear folder"
    expired_time = time.time() * 1000 - 120 * 1000
    for f in os.listdir(CACHE_FOLDER):
        file = os.path.join(CACHE_FOLDER, f)
        if os.stat(file).st_mtime < expired_time:
            os.remove(file)

def findFace(faceDet):
    for i in range(len(FACES_DB)):
        dt = np.subtract(FACES_DB[i], faceDet)
        dt = np.dot(dt, dt)
        if dt < FACE_TRESHOLD :
            return i
    return -1

def regFace(faceDet):
    print("Add face", faceDet)
    FACES_DB.append(faceDet)
    return len(FACES_DB) - 1

def increase_rect(pt1, pt2):
    return (int(pt1[0] * 0.95), int(pt1[1] * 0.9)),\
           (int(math.ceil(pt2[0] * 1.05)), int(math.ceil(pt2[1] * 1.05)))

face_align = None
face_net = None


def process_image(img_origin_size, idx):
    # img_origin_size = cv2.imread("two_person.png")
    # img_origin_size = image = scipy.misc.imread("image.png")
    print "Process {}".format(idx)

    img_origin_size, r = resize(img_origin_size, (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    if FLIP:
        img_origin_size = cv2.flip(img_origin_size, -1)
    img_origin, r = resize(img_origin_size, (SSD_SHAPE_SIZE, SSD_SHAPE_SIZE))
    cache_img = CACHE_FOLDER + "cache.jpg"
    person_out_name = CACHE_FOLDER + "cache_person_" + str(idx)
    person_deepcut_out_name = person_out_name + "_cut.json"

    cv2.imwrite(cache_img, img_origin)
    cv2.imwrite(person_out_name + "_origin.jpg", img_origin_size)

    start = time.clock()

    dets = detector.im_detect([cache_img], None, None, show_timer=True)
    dets = dets[0]

    last_pt1 = None
    last_pt2 = None
    last_cl = None

    persons = []
    for det in dets:
        pt1, pt2, cl = extract_rect(det, img_origin_size.shape[1], img_origin_size.shape[0])
        if cl is None or \
                (last_pt1 == pt1 and last_pt2 == pt2 and last_cl == cl):
            continue
        if cl != "person":
            continue
        print pt1, pt2, cl
        last_pt1 = pt1
        last_pt2 = pt2
        last_cl = cl
        pt1, pt2 = increase_rect(pt1, pt2)

        pose_start = time.clock()
        # crop person
        person_image = img_origin_size[pt1[1]:pt2[1], pt1[0]:pt2[0]]  # cv2.resize(img_origin_size, (224, 224))
        image_copy = person_image.copy()
        if person_image.ndim == 2:
            person_image = np.dstack((person_image, person_image, person_image))
        else:
            person_image = person_image[:, :, ::-1]

        pose = estimate_pose(person_image, model_def, model_bin, [1.])
        pose = activity_detection.filter(pose)
        activity = activity_detection.detect(pose)
        print("pose is {} : time = {} sec".format(activity, time.clock() - pose_start))


        face_start = time.clock()
        # image_face = face_rect(image, pose)
        image_face, r = resize(image_copy, (SSD_SHAPE_SIZE, SSD_SHAPE_SIZE))
        face_rep, face_bb = classifier_webcam.getRep(image_face, args, face_align, face_net)
        face_id = -1
        if len(face_rep) > 0:
            face_rep = face_rep[0]
            face_bb = face_bb[0]
            face_bb = ((int(face_bb.left() / r * .9), int(face_bb.top() / r * .7)), (int(face_bb.right() / r * 1.1), int(face_bb.bottom() / r * 1.2)))
            # face_bb = face_rect(None, pose)
            # min = face_bb[0][1]
            # min_x = (face_bb.left() - min) / 2
            # face_bb = ((face_bb[0][0] - min, face_bb - min), (face_bb.right() - min - min_x,face_bb.bottom() - min))
            face_id = findFace(face_rep)
            if face_id == -1:
                face_id = regFace(face_rep)
        else:
            face_rep = None
            face_bb = None

        print("face (", face_id, " : ", (time.clock() - face_start))
        print("img_preprocessing : ", (time.clock() - start))

        start = time.clock()

        if args.visualize:
            visim = person_image
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
                      [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
                      [0, 0, 0], [255, 255, 255]]
            try:
                for p_idx in range(14):
                    _npcircle(visim,
                              pose[0, p_idx],
                              pose[1, p_idx],
                              4,
                              colors[p_idx],
                              0.0)
                vis_name = person_out_name + '_vis.jpg'
                # cv2.imwrite(vis_name, visim)
            except Exception:
                print "Ignore"

            cv2.rectangle(img_origin_size, pt1, pt2, (255, 0, 0))
            if face_bb != None:
                cv2.rectangle(img_origin_size,
                              (face_bb[0][0] + pt1[0], face_bb[0][1] + pt1[1]),
                              (face_bb[1][0] + pt1[0], face_bb[1][1] + pt1[1]),
                              (255, 255, 0))

        persons.append(
            {'person_id': face_id,
             'poseCoordinats': pose.tolist(),
             'pose_bb' : (pt1, pt2),
             'activity': activity,
             'face_matrix': face_rep.tolist() if face_rep != None else None,
             'face_bb' : face_bb
             })

        # print("visualize : ", (time.clock() - start))

    if len(persons) > 0:
        cv2.imwrite(person_out_name + ".jpg", img_origin_size)
        with open(person_deepcut_out_name, 'wb') as outfile:
            json.dump({'persons': persons}, outfile)
        with open(CACHE_FOLDER + "lastFrame.json", 'wb') as outfile:
            json.dump({'frame': idx}, outfile)
        return True

    return False



if __name__ == '__main__':
    args = parse_args()
    if args.cpu:
        print("With CPU")
        ctx = mx.cpu()
        caffe.set_mode_cpu()
    else:
        print("With GPU : ", args.gpu_id)
        ctx = mx.gpu(args.gpu_id)
        #caffe.set_mode_gpu()
        #caffe.set_device(args.gpu_id)

    detector = get_detector(args.network, args.prefix, args.epoch,
                            args.data_shape,
                            (args.mean_r, args.mean_g, args.mean_b),
                            ctx, args.nms_thresh, args.force_nms)

    #deepcut settings
    model_def = 'deepcut-cnn/models/deepercut/ResNet-152.prototxt'
    model_bin = 'deepcut-cnn/models/deepercut/ResNet-152.caffemodel'
    scales = '1.'

    face_align = openface.AlignDlib("openface/models/dlib/shape_predictor_68_face_landmarks.dat")
    face_net = openface.TorchNeuralNet(
        "openface/models/openface/nn4.small2.v1.t7",
        imgDim=args.imgDim,
        cuda=True)

    # img = cv2.imread("pose_5.jpg")
    # process_image(img, 5)
#
#
# def blabla():
    cap = cv2.VideoCapture("test2.mp4")
    FLIP = True
    # cap = cv2.VideoCapture(0)
    idx = 0

    shutil.rmtree(CACHE_FOLDER)
    os.mkdir(CACHE_FOLDER)
    print("Loaded")

    stream = Subject()
    # stream.buffer_with_count(10)\
    #     .observe_on(Scheduler.new_thread)\
    #     .flat_map(lambda list : Observable.from_(list))\
    #     .subscribe(lambda x : process_image(x[0], x[1]))
    stream.filter(lambda x : x[1] % SKIP_RATE == 0)\
        .subscribe(lambda x: \
                           process_image(x[0], x[1])
                   )

    idx = 0
    while (True):
        ret, img_origin_size = cap.read()
        ret = True
        # print("Video read, ", ret)
        if (img_origin_size is None):
            print "Unable to read image"
            break

        # print "Readed {}".format(idx)
        stream.on_next((img_origin_size, idx))
        # process_image(img_origin_size, idx)
        idx += 1
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

