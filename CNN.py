try:
    import os, subprocess
    gpu_id = subprocess.check_output('gpu_getIDs.sh', shell=True)
    os.environ["THEANO_FLAGS"]='device=gpu%s'%gpu_id
    print(os.environ["THEANO_FLAGS"])
except:
    pass


#import sklearn
import numpy as np
import lasagne
import skimage.transform

from lasagne.utils import floatX

import theano
import theano.tensor as T

import matplotlib.pyplot as plt

import json
import pickle

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer

warning_imports = True
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ReshapeLayer
from lasagne.nonlinearities import softmax
from save_layers import set_all_layers_tags

try:
    from SPP import SpatialPyramidPoolingDNNLayer as SPPolling
except:
    if warning_imports:
        print("-- SPP import failed (probably because you do not have a GPU to run)")

def build_model(input_layer=None,im_size=224,batch_size=None,dropout_value=0.5):
    flip_filter = False
    #warning: if im_size is not 224 the fully connected layer would not work...
    net = {}
    if input_layer!=None:
        net['input'] = input_layer
    else:
        net['input'] = InputLayer((batch_size, 3, im_size, im_size),name='cnn_input')
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=flip_filter,name='conv1_1')
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=flip_filter,name='conv1_2')
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=flip_filter,name='conv2_1')
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=flip_filter,name='conv2_2')
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=flip_filter,name='conv3_1')
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=flip_filter,name='conv3_2')
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=flip_filter,name='conv3_3')
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=flip_filter,name='conv4_1')
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=flip_filter,name='conv4_2')
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=flip_filter,name='conv4_3')
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=flip_filter,name='conv5_1')
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=flip_filter,name='conv5_2')
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=flip_filter,name='conv5_3')
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096,name='fc6')
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=dropout_value)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096,name='fc7')
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=dropout_value)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None,name='fc8')
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    set_all_layers_tags(net['prob'],conv_net=True)
    return net

def add_detbranch(conv5_3,num_boxes,dropout_value=0.5):
    net = {}
    net['boxes'] = InputLayer((None, 5))
    net['crop'] = SPPolling([conv5_3,net['boxes']],pool_dims=7,sp_scale = 1./float(16))
    net['reshape'] = ReshapeLayer(net['crop'], ([0], -1))
    net['det_fc6'] = DenseLayer(net['reshape'], num_units=4096,name='det_fc6')
    net['det_fc6_dropout'] = DropoutLayer(net['det_fc6'], p=dropout_value)
    net['det_fc7'] = DenseLayer(net['det_fc6_dropout'], num_units=4096,name='det_fc7')
    net['det_fc7_dropout'] = DropoutLayer(net['det_fc7'], p=dropout_value)
    net['det_fc8'] = DenseLayer(net['det_fc7_dropout'], num_units=1000, nonlinearity=None,name='det_fc8')
    net['det_prob'] = NonlinearityLayer(net['det_fc8'], softmax)
    return net

def build_model_RCNN(num_boxes,im_size,pool_dims=7,dropout_value=0.5):
    net = {}
    net['input'] = InputLayer((None, 3, im_size, im_size),name='det_image')
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False,name='det_conv1_1')
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False,name='det_conv1_2')
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False,name='det_conv2_1')
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False,name='det_conv2_2')
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False,name='det_conv3_1')
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False,name='det_conv3_2')
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False,name='det_conv3_3')
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False,name='det_conv4_1')
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False,name='det_conv4_2')
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False,name='det_conv4_3')
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False,name='det_conv5_1')
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False,name='det_conv5_2')
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False,name='det_conv5_3')
#detection branch
    net['boxes'] = InputLayer((None, 5),name='det_boxes')
    net['crop'] = SPPolling([net['conv5_3'],net['boxes']],pool_dims=pool_dims,sp_scale = 1./float(16))
    net['reshape'] = ReshapeLayer(net['crop'], ([0], -1))
    net['fc6'] = DenseLayer(net['reshape'], num_units=4096,name='det_fc6')
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=dropout_value)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096,name='det_fc7')
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=dropout_value)
    net['cls_score'] = DenseLayer(net['fc7_dropout'], num_units=1000, nonlinearity=None,name='det_fc8')
    net['prob'] = NonlinearityLayer(net['cls_score'], softmax)
    set_all_layers_tags(net['prob'],conv_net2=True)
    return net


MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

def prep_image(im,im_size=224):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    # Resize so smallest dim = 224, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (im_size, w*im_size/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*im_size/w, im_size), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-im_size/2:h//2+im_size/2, w//2-im_size/2:w//2+im_size/2]

    rawim = np.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return floatX(im[np.newaxis])

PIXEL_MEANS = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
try:
    import cv2 as cv
except:
    print("-- cv2 could not be imported")

def dedupboxes(boxes,dedup_val=0.0625):
    if dedup_val > 0:
        #v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        v = np.array([1, 1e3, 1e6, 1e9])
        hashes1 = np.round(boxes * dedup_val).dot(v)
        _, index1, inv_index = np.unique(hashes1, return_index=True,
                                        return_inverse=True)
        boxes = boxes[index1, :]
    return boxes

#def prep_image_RCNN_old(orig_img, pixel_means = PIXEL_MEANS, max_size=1000, scale=600):
#    img = orig_img.astype(np.float32, copy=True)
#    if len(img.shape) == 2:
#        img = img[:, :, np.newaxis]
#        img = np.repeat(img, 3, axis=2)
#    img -= pixel_means
#    im_size_min = np.min(img.shape[0:2])
#    im_size_max = np.max(img.shape[0:2])
#    im_scale = float(scale) / float(im_size_min)
#    if np.rint(im_scale * im_size_max) > max_size:
#        im_scale = float(max_size) / float(im_size_max)
#    img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
#                    interpolation=cv.INTER_LINEAR)
#
#    return img.transpose([2, 0, 1]).astype(np.float32), im_scale

def prep_image_RCNN(orig_img, pixel_means = PIXEL_MEANS, max_size=1000, scale=600):
    img = orig_img.astype(np.float32, copy=True)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)
    img= img[:,:,::-1]
    img -= pixel_means
    im_size_min = np.min(img.shape[0:2])
    im_size_max = np.max(img.shape[0:2])
    im_scale = float(scale) / float(im_size_min)
    if np.rint(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv.INTER_LINEAR)
    return img.transpose([2, 0, 1]).astype(np.float32), im_scale


import pylab
def plot_boxes(boxes):
    for idb in range(boxes.shape[0]):
        x0 = boxes[idb,0]
        y0 = boxes[idb,1]
        x1 = boxes[idb,2]
        y1 = boxes[idb,3]
        pylab.plot([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0],lw=3)


if __name__ == "__main__":

    vgg16net = build_model()
    cnn_layers = vgg16net
    cnn_input_var = cnn_layers['input'].input_var
    cnn_conv_feature_layer = cnn_layers['conv5_2']
    cnn_feature_layer = cnn_layers['fc7']
    cnn_output_layer = cnn_layers['prob']

    get_cnn_features = theano.function([cnn_input_var], lasagne.layers.get_output(cnn_feature_layer))
    get_cnn_conv_features = theano.function([cnn_input_var], lasagne.layers.get_output(cnn_conv_feature_layer))

    model_param_values = pickle.load(open('data/vgg16.pkl'))['param values']
    lasagne.layers.set_all_param_values(cnn_output_layer, model_param_values)



