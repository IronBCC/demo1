


import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)

def import_module(path):
    sys.path.insert(0, path)

import openface
#import pose


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

import scipy

from collections import namedtuple

from mxnet.io import DataBatch

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
    parser.add_argument('--verbose', dest='verbose', type=bool, default=True,
                        help='show detection time')
    parser.add_argument('--classifierModel', dest='classifierModel', type=str, default="openface/models/openface/celeb-classifier.nn4.small2.v1.pkl",
                        help='show detection time')
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
    return cv2.resize(img, size)

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

    # align = openface.AlignDlib("openface/models/dlib/shape_predictor_68_face_landmarks.dat")
    # net = openface.TorchNeuralNet(
    #     "openface/models/openface/nn4.small2.v1.t7",
    #     imgDim=96,
    #     cuda=not args.cpu)

    cap = cv2.VideoCapture("test2.mp4")
    idx = 0

    shutil.rmtree(CACHE_FOLDER)
    os.mkdir(CACHE_FOLDER)
    print("Loaded")
    while (True):
        ret, img_origin_size = cap.read()
        # img_origin_size = cv2.imread("two_person.png")
        # img_origin_size = image = scipy.misc.imread("image.png")
        ret= True
        print("Video read, ", ret)
        if(img_origin_size is None):
            print "Unable to read image"
            break

        img_origin_size = resize(img_origin_size, (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        img_origin_size = cv2.flip(img_origin_size, -1)
        img_origin = resize(img_origin_size, (SSD_SHAPE_SIZE, SSD_SHAPE_SIZE))
        cache_img = CACHE_FOLDER + "cache.jpg"
        person_out_name = CACHE_FOLDER + "cache_person_" + str(idx)
        person_deepcut_out_name = person_out_name + "_cut.json"

        cv2.imwrite(cache_img, img_origin)


        start = time.clock()

        dets = detector.im_detect([cache_img], None, None, show_timer=True)
        dets = dets[0]

        last_pt1 = None
        last_pt2 = None
        last_cl = None
        for det in dets:
            pt1, pt2, cl = extract_rect(det, img_origin_size.shape[1], img_origin_size.shape[0])
            if cl is None or \
                    (last_pt1 == pt1 and last_pt2 == pt2 and last_cl == cl):
                continue
            if cl != "person" :
                continue

            print pt1, pt2, cl
            last_pt1 = pt1
            last_pt2 = pt2
            last_cl = cl
            #TODO: path only people squeare

            pose_start = time.clock()
            #crop person
            image = img_origin_size[pt1[1]:pt2[1], pt1[0]:pt2[0]]#cv2.resize(img_origin_size, (224, 224))

            # classifier_webcam.infer(image, args, align, net)

            if image.ndim == 2:
                image = np.dstack((image, image, image))
            else:
                image = image[:, :, ::-1]
            print(" pose progress ", image.shape)
            pose = estimate_pose(image, model_def, model_bin, [1.])

            print("pose : ", (time.clock() - pose_start))
            print("img_preprocessing : ", (time.clock() - start))
            start = time.clock()

            if args.visualize:
                visim = image
                colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
                          [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
                          [0, 0, 0], [255, 255, 255]]
                try :
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

            with open(person_deepcut_out_name, 'wb') as outfile:
                json.dump(pose.tolist(), outfile)
            cv2.imwrite(person_out_name + ".jpg", img_origin_size)

            print("visualize : ", (time.clock() - start))
        idx += 1
        # time.sleep(1)
        # Display the resulting frame
        # cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
