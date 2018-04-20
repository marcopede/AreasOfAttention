# coding: utf-8
# vim:fdm=marker
# # Image Captioning with LSTM
#

import time

import pickle
import random
import numpy as np
import argparse
import sys
import CNN
import pylab
from scipy.io import loadmat
from bcolors import bcolors
from flickr_prepareData import load_flickr

try:
    import os
    import subprocess
    gpu_id = subprocess.check_output('gpu_getIDs.sh', shell=True)
    os.environ["THEANO_FLAGS"] = 'device=gpu%s' % gpu_id
    print(os.environ["THEANO_FLAGS"])
except:
    pass

import theano
#theano.config.profile = True
# theano.config.optimizer='None'#'fast_compile'
# theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'#'fast_compile'

import theano.tensor as T
import lasagne

from collections import Counter
from lasagne.utils import floatX
from save_layers import get_param_dict_tied, set_param_dict, set_all_layers_tags, check_init
#from lasagne.layers import SoftmaxLayer
from TProd3 import MySoftmaxLayer
from Transformer import TranslateLayer
import resnet_CNN

# can use matplotlib without having a display
import matplotlib
matplotlib.use('Agg')
starting_time = time.time()
sys.setrecursionlimit(100000)
theano.config.optimizer = 'None'

CFG = {
    'FILE_NAME': './data/test',
    'SEQUENCE_LENGTH': 22,  # 32 #carefull, it seems that loses in B4 with shorter sentences
    'BATCH_SIZE': 10,
    'CNN_FEATURE_SIZE': 4096,  # 1000
    'EMBEDDING_SIZE': 512,
    'EMBEDDING_WORDS': 512,
    'NUM_REGIONS': 14 * 14,
    'REGION_SIZE': 512,
    'CONV_NORMALIZED': True,  # False#True
    # '../data/lstm_coco_tensor_tensor-reducedw_100000.pkl', #start from scratch
    'START_FROM': '',
    'LR': 0.001,  # note that now defalut is 10 times bigger than before...
    'MODE': 'normal',
    'REG_H': 0,
    #'MAX_ITER': 2000000,
    'IM_SIZE': 224,
    # parameters set by default to false because from the command line they can be set to true
    'CONV_REDUCED': False,
    'RESTART': False,
    'TENSOR_RECTIFY': False,
    'CNN_MODEL': 'vgg',
    'RESNET_LAYER': 'prob',
    #    mode = 'normal'
    #    mode = 'tensor'
    #    mode = 'tensor-reducedw'
    #    mode = 'tensor-removedWrw'
}

# 1 for image, 1 for start token, 1 for end token
MAX_SENTENCE_LENGTH = CFG['SEQUENCE_LENGTH'] - 3
MAX_BOXES = 2000


def calc_cross_ent(net_output, mask, targets):
    # Helper function to calculate the cross entropy error
    preds = T.reshape(net_output, (-1, len(vocab)))
    targets = T.flatten(targets)

    # Only non masked classes are returned.
    cost = T.nnet.categorical_crossentropy(
        preds, targets)[T.flatten(mask).nonzero()]
    return cost


def get_Regions_cond_words_slow(X):
    Xw = X.sum(2)  # marginalize regions
    X1 = Xw.reshape((-1, Xw.shape[2]))
    m = X1.argmax(1)
    X2 = X.reshape((X1.shape[0], X.shape[2], -1))
    X3 = X2[T.arange(X1.shape[0]), :, m].reshape((X.shape[0], X.shape[1], -1))
    return X3


def get_Regions_cond_words(X):
    Xw = X.sum(2)  # marginalize regions
    X1 = Xw.reshape((-1, Xw.shape[2]))
    m = T.cast(X1.argmax(1), 'int32')
    X2 = X.reshape((X1.shape[0], X.shape[2], -1))
    X3 = X2[T.arange(X1.shape[0], dtype='int32'), :, m].reshape(
        (X.shape[0], X.shape[1], -1))
    return X3


def flip_boxes(boxes, img_shape_x):
    """
        box(image, x0,y0,x1,y1)
        img_shape(dimy,dimx)
        return flipped box
    """
    flipped_box = np.zeros(boxes.shape, dtype=np.float32)
    flipped_box[:, 0] = boxes[:, 0]
    flipped_box[:, 1] = img_shape_x - boxes[:, 3]
    flipped_box[:, 2] = boxes[:, 2]
    flipped_box[:, 3] = img_shape_x - boxes[:, 1]
    flipped_box[:, 4] = boxes[:, 4]
    return flipped_box


from scipy.ndimage.interpolation import rotate


def rotate_image(img, dangle=5):
    dest = np.zeros((4, img.shape[0], img.shape[1], img.shape[2]), img.dtype)
    dest[0] = rotate(img, dangle, axes=(1, 2), reshape=False)
    dest[1] = rotate(img, -dangle, axes=(1, 2), reshape=False)
    dest[2] = rotate(img, 2 * dangle, axes=(1, 2), reshape=False)
    dest[3] = rotate(img, -2 * dangle, axes=(1, 2), reshape=False)
    return dest


def deconvert(val, CFG):
    # {{{
    if CFG['DATASET'] == 'coco':
        sample = PrepareData.deconvertMat(val['image'][0][0][0])
    else:
        sample = PrepareData.deconvert(val['image'])
    return sample
    # }}}


import PrepareData
import StringIO
showdebug = False


def batch_gen(CFG, db, batch_size, im_scale, word_to_index, shuffle, max_im_size=-1, shuffle_boxes=False, start_from=0, force_missing_boxes=False, use_flip=None, use_rotations=False):
    get_boxes = (CFG['DATASET'] == 'coco')
    val = db.iteritems().next()[1]

    count_rot = 4 if use_rotations else 0
    sample = deconvert(val, CFG)
    _im = CNN.prep_image(sample)
    imgs = floatX(
        np.zeros((batch_size, _im.shape[1], _im.shape[2], _im.shape[3])))
    if max_im_size == -1:
        max_im_size = int(im_scale * 1.5)
    if im_scale != -1:
        imgs2 = floatX(np.zeros((batch_size, 3, max_im_size, max_im_size)))
        imgs2[...] = CNN.PIXEL_MEANS[np.newaxis, :, np.newaxis, np.newaxis]
    else:
        imgs2 = floatX(np.zeros((batch_size, 3, 1, 1)))
    if get_boxes:
        x_boxes = floatX(-np.ones((batch_size, CFG['NUM_REGIONS'], 5)))
    else:
        x_boxes = None
    x_sentence = np.zeros(
        (batch_size * 5, CFG['SEQUENCE_LENGTH'] - 1), dtype='int32')
    y_sentence = np.zeros(
        (batch_size * 5, CFG['SEQUENCE_LENGTH']), dtype='int32')
    mask = np.zeros((batch_size * 5, CFG['SEQUENCE_LENGTH']), dtype='bool')
    count = 0
    ks = []
    dbkeys = db.keys()
    # Random operation, depending on the seed
    if shuffle:
        random.shuffle(dbkeys)
    for k in dbkeys[start_from:]:
        if use_flip == 'sample':
            flip = np.random.randint(2)
        v = db[k]
        #im_sample = PrepareData.deconvertMat(v['image'][0][0][0])
        im_sample = deconvert(v, CFG)
        imgs[count] = CNN.prep_image(im_sample)[0]
        if use_rotations:
            imgs[count + 1:count + count_rot + 1] = rotate_image(imgs[count])
        if use_flip == 'both':
            imgs[count + count_rot + 1] = imgs[count, :, :, ::-1]
            if use_rotations:
                imgs[count + count_rot + 2:count + 2 *
                     count_rot + 2] = rotate_image(imgs[count])
        elif use_flip == 'sample' and flip:
            imgs[count] = imgs[count, :, :, ::-1]
        if im_scale != -1:
            if use_rotations:
                print('Error, rotations not implemented for proposals!')
                sys.exit()
            imgs2[count] = CNN.PIXEL_MEANS[:, np.newaxis, np.newaxis]
            newimgs2, scale = CNN.prep_image_RCNN(
                im_sample, scale=im_scale, max_size=max_im_size)
            imgs2[count, :, :newimgs2.shape[1], :newimgs2.shape[2]] = newimgs2
            if use_flip == 'both':
                imgs2[count + 1, :, :newimgs2.shape[1],
                      :newimgs2.shape[2]] = newimgs2[:, :, ::-1]
            elif use_flip == 'sample' and flip:
                imgs2[count, :, :newimgs2.shape[1],
                      :newimgs2.shape[2]] = newimgs2[:, :, ::-1]
            if get_boxes:
                num_boxes = min(CFG['NUM_REGIONS'], v['boxes'][0][0].shape[0])
                boxes = v['boxes'][0][0]
                order_boxes = np.arange(len(boxes))
                if shuffle_boxes:
                    order_boxes.random.shuffle()
                scaled_boxes = boxes[order_boxes] * scale
                dedup_boxes = scaled_boxes[:num_boxes]
                num_boxes = min(num_boxes, len(dedup_boxes))
                x_boxes[count, :num_boxes] = np.concatenate(
                    (count * np.ones((num_boxes, 1), dtype='int32'), dedup_boxes), 1)
                if use_flip == 'both':
                    x_boxes[count + 1, :num_boxes] = flip_boxes(
                        x_boxes[count, :num_boxes], newimgs2.shape[2])
                elif use_flip == 'sample' and flip:
                    x_boxes[count, :num_boxes] = flip_boxes(
                        x_boxes[count, :num_boxes], newimgs2.shape[2])
            if 0:
                import pylab
                pylab.figure()
                print x_boxes[count]
                CNN.plot_boxes(x_boxes[count][:, 1:])
                pylab.draw()
                pylab.show()
                import pdb
                pdb.set_trace()

            if get_boxes:
                if num_boxes < CFG['NUM_REGIONS'] and force_missing_boxes == False:
                    print('Warning: less boxes than expected, skipping sample!')
                    continue
        ks.append(k)
        if use_rotations:
            ks += [k] * 4
        if use_flip == 'both':
            ks.append(k)
            if use_rotations:
                ks += [k] * 4
        if CFG['DATASET'] == 'coco':
            sentence = v['caption'][0][0]
        else:
            sentence = v['caption']
        for ss in range(5):
            i = 0
            if CFG['CLEAN_MASKS']:
                mask[count * 5 + ss, :] = False
                y_sentence[count * 5 + ss, :] = word_to_index['#END#']
                x_sentence[count * 5 + ss, :] = word_to_index['#END#']
                if use_rotations:
                    mask[(count + 1) * 5 + ss:(count + count_rot + 1)
                         * 5 + ss, :] = False
                    y_sentence[(count + 1) * 5 + ss:(count + count_rot + 1)
                               * 5 + ss, :] = word_to_index['#END#']
                    x_sentence[(count + 1) * 5 + ss:(count + count_rot + 1)
                               * 5 + ss, :] = word_to_index['#END#']
                if use_flip == 'both':
                    mask[(count + count_rot + 1) * 5 + ss, :] = False
                    y_sentence[(count + count_rot + 1) * 5 +
                               ss, :] = word_to_index['#END#']
                    x_sentence[(count + count_rot + 1) * 5 +
                               ss, :] = word_to_index['#END#']
                    if use_rotations:
                        mask[(count + count_rot + 2) * 5 + ss:(count +
                                                               2 * count_rot + 3) * 5 + ss, :] = False
                        y_sentence[(count + count_rot + 2) * 5 + ss:(count + 2 *
                                                                     count_rot + 3) * 5 + ss, :] = word_to_index['#END#']
                        x_sentence[(count + count_rot + 2) * 5 + ss:(count + 2 *
                                                                     count_rot + 3) * 5 + ss, :] = word_to_index['#END#']
            sent = PrepareData.tokenize(sentence[ss])
            for word in (['#START#'] + sent + ['#END#'])[:MAX_SENTENCE_LENGTH]:
                if word in word_to_index:
                    mask[count * 5 + ss, i] = True
                    y_sentence[count * 5 + ss, i] = word_to_index[word]
                    x_sentence[count * 5 + ss, i] = word_to_index[word]
                    if use_rotations:
                        mask[(count + 1) * 5 + ss:(count + count_rot + 1)
                             * 5 + ss, :] = True
                        y_sentence[(count + 1) * 5 + ss:(count + count_rot + 1)
                                   * 5 + ss, :] = word_to_index[word]
                        x_sentence[(count + 1) * 5 + ss:(count + count_rot + 1)
                                   * 5 + ss, :] = word_to_index[word]
                    if use_flip == 'both':
                        mask[(count + count_rot + 1) * 5 + ss, i] = True
                        y_sentence[(count + count_rot + 1) * 5 +
                                   ss, i] = word_to_index[word]
                        x_sentence[(count + count_rot + 1) * 5 +
                                   ss, i] = word_to_index[word]
                        if use_rotations:
                            mask[(count_count_rot + 2) * 5 +
                                 ss:(count + count_rot + 2) * 5 + ss, :] = True
                            y_sentence[(count + count_rot + 2) * 5 + ss:(count +
                                                                         count_rot + 2) * 5 + ss, :] = word_to_index[word]
                            x_sentence[(count + count_rot + 2) * 5 + ss:(count +
                                                                         count_rot + 2) * 5 + ss, :] = word_to_index[word]
                    i += 1
                else:
                    y_sentence[count * 5 + ss, i] = word_to_index['#NAW#']
                    x_sentence[count * 5 + ss, i] = word_to_index['#NAW#']
                    if use_rotations:
                        y_sentence[(count + 1) * 5 + ss:(count + count_rot + 1)
                                   * 5 + ss, i] = word_to_index['#NAW#']
                        x_sentence[(count + 1) * 5 + ss:(count + count_rot + 1)
                                   * 5 + ss, i] = word_to_index['#NAW#']
                    if use_flip == 'both':
                        y_sentence[(count + count_rot + 1) * 5 +
                                   ss, i] = word_to_index['#NAW#']
                        x_sentence[(count + count_rot + 1) * 5 +
                                   ss, i] = word_to_index['#NAW#']
                        if use_rotations:
                            y_sentence[(count + count_rot + 2) * 5 + ss:(count +
                                                                         count_rot + 2) * 5 + ss, i] = word_to_index['#NAW#']
                            x_sentence[(count + count_rot + 2) * 5 + ss:(count +
                                                                         count_rot + 2) * 5 + ss, i] = word_to_index['#NAW#']
                    i += 1
            mask[count * 5 + ss, 0] = False
            if use_rotations:
                mask[(count + 1) * 5 + ss:(count + count_rot + 1)
                     * 5 + ss, 0] = False
            if use_flip == 'both':
                mask[(count + count_rot + 1) * 5 + ss, 0] = False
                if use_rotations:
                    mask[(count + count_rot + 2) * 5 +
                         ss:(count + count_rot + 2) * 5 + ss, 0] = False
            if showdebug:
                print sent
        if showdebug:
            import pylab
            pylab.figure(1)
            pylab.clf()
            pylab.imshow(PrepareData.deconvert(v, CFG))
            pylab.draw()
            pylab.show()
            print mask[count * 5:(count + 1) * 5]
            import pdb
            pdb.set_trace()
        count += 1
        if use_rotations:
            count += count_rot
        if use_flip == 'both':
            count += 1
            if use_rotations:
                count += count_rot
        if count >= batch_size:
            count = 0
            assert len(ks) == len(imgs)
            # imgs=imgs[:,:,:,::-1]# to remove...
            if get_boxes:
                yield (ks, imgs, imgs2, x_sentence, y_sentence, mask, x_boxes.reshape((batch_size * CFG['NUM_REGIONS'], 5)))
            else:
                yield (ks, imgs, imgs2, x_sentence, y_sentence, mask, None)
            ks = []
    while len(ks) < CFG['BATCH_SIZE']:
        ks.append(-1)
    if get_boxes:
        yield (ks, imgs, imgs2, x_sentence, y_sentence, mask, x_boxes.reshape((batch_size * CFG['NUM_REGIONS'], 5)))
    else:
        yield (ks, imgs, imgs2, x_sentence, y_sentence, mask, None)


from lasagne.layers import InputLayer, EmbeddingLayer, DenseLayer, ExpressionLayer, \
        ReshapeLayer, DimshuffleLayer, ConcatLayer, NonlinearityLayer, DropoutLayer, \
        TransformerLayer, ExpressionLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer

from Transformer import MultiTransformerLayer
from lasagne.layers import dropout
from lasagne.nonlinearities import softmax

# basic model


def build_seq_input_net(CFG, vocab, l_input_sequence, l_input_img):
    l_sentence_embedding = EmbeddingLayer(l_input_sentence,
                                          input_size=len(vocab),
                                          output_size=CFG['EMBEDDING_SIZE'],
                                          name='l_sentence_embedding'
                                          )
    # cnn embedding changes the dimensionality of the representation to EMBEDDING_SIZE,
    # and reshapes to add the time dimension - final dim (BATCH_SIZE, 1, EMBEDDING_SIZE)
    import CNN
    # Chose the CNN (Resnet or VGG)
    if CFG['CNN_MODEL'] == 'vgg':
        print("Using VGG as a CNN.")
        vgg16 = CNN.build_model(
            input_layer=l_input_img, batch_size=CFG['BATCH_SIZE'], dropout_value=CFG['RNN_DROPOUT'])
        print "Loading pretrained VGG16 parameters"
        model_param_values = pickle.load(open('vgg16.pkl'))['param values']
        lasagne.layers.set_all_param_values(vgg16['prob'], model_param_values)
        #l_input_img = vgg16['input']
        l_input_cnn = vgg16['fc7_dropout']
    elif CFG['CNN_MODEL'] == 'resnet':
        print("Using Resnet-50 as a CNN.")

        resnet50 = resnet_CNN.build_model(
            input_layer=l_input_img, batch_size=CFG['BATCH_SIZE'])
        print "Loading pretrained Resnet-50 parameters"
        # You can use this format to store other things for best effort
        model_param_values = pickle.load(open('resnet50.pkl'))['param values']

        from save_layers import add_names_layers_and_params
        add_names_layers_and_params(resnet50)
        set_param_dict(resnet50['prob'], model_param_values,
                       prefix='', show_layers=False, rlax=False)
        l_input_img = resnet50['input']

        if CFG['RESNET_LAYER'] == 'prob':
            l_input_cnn = resnet50['prob']
        elif CFG['RESNET_LAYER'] == 'pool5':
            l_input_cnn = resnet50['pool5']
        else:
            print("Layer not supported for resnet")
            raise ValueError()
    else:
        print("Unknown CNN selected")

    if CFG['START_NORMALIZED'] == 1:
        l_input_cnn = ExpressionLayer(
            l_input_cnn, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
    elif CFG['START_NORMALIZED'] == 2:
        l_input_cnn = ExpressionLayer(l_input_cnn, lambda X: X / T.sqrt(
            T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
    else:
        l_input_cnn = ExpressionLayer(
            l_input_cnn, lambda X: X * 0.01, output_shape='auto')
    l_cnn_embedding = DenseLayer(l_input_cnn, num_units=CFG['EMBEDDING_SIZE'],
                                 nonlinearity=lasagne.nonlinearities.identity, name='l_cnn_embedding')
    l_cnn_embedding = ExpressionLayer(
        l_cnn_embedding, lambda X: theano.tensor.extra_ops.repeat(X, 5, axis=0), output_shape='auto')
    l_cnn_embedding2 = ReshapeLayer(
        l_cnn_embedding, ([0], 1, [1]), name='l_cnn_embedding2')

    # the two are concatenated to form the RNN input with dim (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE)
    l_rnn_input = ConcatLayer(
        [l_cnn_embedding2, l_sentence_embedding], name='l_rnn_input')
    l_dropout_input = DropoutLayer(
        l_rnn_input, p=CFG['RNN_DROPOUT'], name='l_dropout_input')
    # set the flag seq_input
    set_all_layers_tags(l_dropout_input, seq_input=True)

    if CFG['CNN_MODEL'] == 'vgg':
        if CFG['TRANS_USE_PRETRAINED']:
            return l_dropout_input, vgg16['conv5_2'], vgg16['conv5_3']
        else:
            return l_dropout_input, vgg16['conv5_3']
    if CFG['CNN_MODEL'] == 'resnet':
        if CFG['TRANS_USE_PRETRAINED']:
            return l_dropout_input, resnet50['res4e_relu'], resnet50['res4f_branch2b']
        else:
            return l_dropout_input, resnet50['res4f_relu']


def build_loc_net(CFG, l_gru, l_input_regions_cell, expand, size_im, size_kernel, stride, zoom, W, b, name=''):
    #
    l_input_regions_reduced = lasagne.layers.ExpressionLayer(
        l_input_regions_cell, lambda X: X[:, :, ::stride, ::stride], output_shape='auto')
    num_prop = ((size_im[0] + stride - 1) / stride) * \
        ((size_im[1] + stride - 1) / stride)
    l_input_regions2 = ReshapeLayer(
        l_input_regions_reduced, (expand * CFG['BATCH_SIZE'], CFG['REGION_SIZE'], num_prop))
    l_input_loc = DimshuffleLayer(l_input_regions2, (0, 2, 1))
    l_input_loc = ReshapeLayer(
        l_input_loc, (expand * CFG['BATCH_SIZE'] * num_prop, CFG['REGION_SIZE']))
    if CFG['TRANS_USE_STATE']:
        assert(expand == 5)  # in this case we cannot save memory
        l_input_state = lasagne.layers.ExpressionLayer(
            l_gru, lambda X: theano.tensor.extra_ops.repeat(X, num_prop, axis=0), output_shape='auto')
        l_input_loc = ConcatLayer((l_input_loc, l_input_state), axis=1)
    # normalization
    if CFG['CONV_NORMALIZED'] == 1:
        l_input_loc = lasagne.layers.ExpressionLayer(
            l_input_loc, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
    elif CFG['CONV_NORMALIZED'] == 2:
        l_input_loc = lasagne.layers.ExpressionLayer(l_input_loc, lambda X: X / T.sqrt(
            T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
    else:
        l_input_loc = lasagne.layers.ExpressionLayer(
            l_input_loc, lambda X: X * 0.01, output_shape='auto')

    if CFG['TRANS_LOCNET_DROPOUT'] > 0.:
        l_input_loc = dropout(l_input_loc, p=CFG['TRANS_LOCNET_DROPOUT'])
    if CFG['TRANS_LOCNET'] == 0:  # use relu
        l_input_loc = DenseLayer(
            l_input_loc, num_units=512, W=W, b=b, name='l_input_loc' + name)
    elif CFG['TRANS_LOCNET'] == 1:  # linear to avoid rotations
        l_input_loc = DenseLayer(
            l_input_loc, num_units=512, W=W, b=b, nonlinearity=None, name='l_input_loc' + name)
    # add concatenation with state
    if CFG['TRANS_NOROT']:
        l_loc = DenseLayer(l_input_loc, num_units=4, W=W, b=b,
                           nonlinearity=None, name='l_loc' + name)
        l_loc = ExpressionLayer(l_loc, lambda X: T.concatenate([X[:, :1], T.zeros((expand * CFG['BATCH_SIZE'] * num_prop, 1), dtype=np.float32), X[:, 1:2], T.zeros(
            (expand * CFG['BATCH_SIZE'] * num_prop, 1), dtype=np.float32), X[:, 2:4]], axis=1), output_shape='auto')
    else:
        l_loc = DenseLayer(l_input_loc, num_units=6, W=W, b=b,
                           nonlinearity=None, name='l_loc' + name)
    # even if the model has been trained with rotations, force the model to not use them
    if CFG.has_key('TRANS_FORCE_NOROT') and CFG['TRANS_FORCE_NOROT']:
        l_loc = ExpressionLayer(l_loc, lambda X: T.concatenate([X[:, :1], T.zeros(
            (expand * CFG['BATCH_SIZE'] * num_prop, 1)), X[:, 2:3], T.zeros((expand * CFG['BATCH_SIZE'] * num_prop, 1)), X[:, 4:6]], axis=1), output_shape='auto')
    l_tr_loc = TranslateLayer(l_loc, (size_im[0], size_im[1]), (
        size_kernel[0], size_kernel[1]), stride=stride, zoom=zoom)  # returns (batch,num_regions,6)
    trainable = True
    slowlearn = False
    reglearn = False
    locnet = True
    if CFG['TRANS_SLOWLEARN']:
        trainable = False
        slowlearn = True
    if CFG['TRANS_REGLEARN'] > 0:
        trainable = True
        reglearn = True
    if CFG['TRANS_NOLEARN']:
        trainable = False
    from save_layers import set_all_layers_tags
    set_all_layers_tags(l_tr_loc, treat_as_input=[
                        l_gru, l_input_regions2], trainable=trainable, slowlearn=slowlearn, reglearn=reglearn, locnet=locnet)
    return num_prop, l_tr_loc


def check_overlap(dbtrain, dbval):
    if check_overlap:
        for k in dbval:
            if dbtrain.has_key(k):
                print "Error: overlap between the two sets!"
                sys.exit()
        print('No overlap between the two sets!')


def load_coco(path='coco/', no_train=False, small=False, add_validation=False, check_overlap=False, new_test=False, karpathy_split=False):
    # {{{
    t = time.time()
    # not in the parameters because it shoudl not change between training and validation
    add_validation_size = 10000
    add_test_size = 5000

    if small:
        dbval = loadmat(path + 'dbval_small.mat')
    else:
        dbval = loadmat(path + 'dbval.mat')
        dbtrain = loadmat(path + 'dbtrain.mat')

    del dbval['__version__']
    del dbval['__header__']
    del dbval['__globals__']

    if no_train and not new_test:
        # if no_train splits the validation in val and test
        dbvalkeys = dbval.keys()
        dbvalord = np.load(open('dbval_order'))
        dbvalkeys = [dbvalkeys[x] for x in dbvalord]
        testkeys = dbvalkeys[:add_test_size]
        dbtest = {key: dbval[key] for key in testkeys}
        newvalkeys = dbvalkeys[add_test_size:]
        valdb = {key: dbval[key] for key in newvalkeys}
        # if add test, validation is reduced to 5000
        add_validation_size = 5000

    if add_validation or new_test:
        dbvalkeys = dbval.keys()
        dbvalkeys.sort()
        newvalkeys = dbvalkeys[:add_validation_size]
        newvaldb = {key: dbval[key] for key in newvalkeys}
        if no_train and not new_test:
            dbval = newvaldb

    if new_test:  # the old test was overlapping with the additional data
        newvalkeys1 = dbvalkeys[:add_validation_size / 2]
        newvalkeys2 = dbvalkeys[add_validation_size / 2:add_validation_size]
        newdbval = {key: dbval[key] for key in newvalkeys1}
        newdbtest = {key: dbval[key] for key in newvalkeys2}
        return newdbval, newdbtest

    if karpathy_split:
        print("Using Karpathy splits...")
        karpathy_path = './Karpathy_splits/'

        import json
        karp_dbval_info = json.load(open(karpathy_path + 'val_data.json'))
        karp_dbval_keys = [elem['id'] for elem in karp_dbval_info]
        karp_dbtest_info = json.load(open(karpathy_path + 'test_data.json'))
        karp_dbtest_keys = [elem['id'] for elem in karp_dbtest_info]

        karp_dbval = {key: dbval[key] for key in karp_dbval_keys}
        karp_dbtest = {key: dbval[key] for key in karp_dbtest_keys}
        return karp_dbval, karp_dbtest

    if no_train and not new_test and not karpathy_split:
        return dbval, dbtest

    if small:
        dbtrain = loadmat(path + 'dbtrain_small.mat')
    else:
        dbtrain = loadmat(path + 'dbtrain.mat')

    del dbtrain['__version__']
    del dbtrain['__header__']
    del dbtrain['__globals__']

    print("Dataset Loaded in %d sec" % (time.time() - t))
    if add_validation:
        addtrainkeys = dbvalkeys[add_validation_size:]
        addtraindb = {key: dbval[key] for key in addtrainkeys}
        dbval = newvaldb
        dbtrain.update(addtraindb)

    if not karpathy_split:
        if check_overlap:
            for k in dbval:
                if dbtrain.has_key(k):
                    print "Error: overlap between training and validation data!"
                    sys.exit()
            print('No overlap between training and validation data!')

    return dbtrain, dbval
    # }}}


def load_config(CFG, CFG1):
    if d.has_key('loss train'):
        loss_tr = d['loss train']
        loss_val = d['loss val']
    error = False
    # check that the configuretion is exactly the same!
    for key, value in CFG.items():
        if key == 'RESTART':  # compatibility with old CFG
            continue
        if not CFG1.has_key(key):
            print "Warning: MISSING PARAMETERS"
            print key
            error = True
            continue
        if value != CFG1[key]:
            print 'Warning: DIFFERENT PARAMETRS!!!'
            print 'Current CFG[', key, ']=', value
            print 'Loaded CFG[', key, ']=', CFG1[key]
            error = True
    if error and not params['force_load'] and not params['force_current']:
        raise ValueError(
            'Warning found, use force_load if you want to run it with the loaded parameters, or force_current for using the current command line parameters')
    if params['force_load']:
        for key, value in CFG1.items():
            CFG[key] = CFG1[key]
    return CFG, loss_tr, loss_val


def save_epoc(epoc, updates, param_values, vocab, word_to_index, index_to_word, CFG, loss_tr, loss_val, partial=False):
    adam_values = [item.get_value() for item in updates]
    d = {'param values': param_values,
         'vocab': vocab,
         'word_to_index': word_to_index,
         'index_to_word': index_to_word,
         'config': CFG,
         'loss train': loss_tr,
         'loss val': loss_val,
         'adam values': adam_values
         }
    import os
    if not os.path.exists(CFG['FILE_NAME']):
        os.makedirs(CFG['FILE_NAME'])
    save_time = time.time()
    save_name = '%s/%s_%d.pkl' % (CFG['FILE_NAME'], CFG['MODE'], epoc + 1)
    if partial != False:
        save_name = save_name + '.partial'
        d['partial'] = partial
    print "Saving", save_name
    pickle.dump(d, open(save_name, 'w'), protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved in :{}s'.format(time.time() - save_time))
    return save_name


if __name__ == "__main__":

    # Parser {{{ 1
    # overwrite parameters form command line
    parser = argparse.ArgumentParser()

    # basic options
    parser.add_argument('filename', default='', nargs='?',
                        help='pkl file where to save the model')
    parser.add_argument('-m', '--mode', dest='mode', type=int, default=-1,
                        help='Network structure: (0):normal,(1):tensor,(2):tensor-reducedw,(3)tensor-removedWrw,(4)tensor-feedback (5)tensor-feedback2 (6)feedbackH (not working yet:()')
    parser.add_argument('--dataset', type=str,
                        default='coco', help='flickr or coco')
    parser.add_argument('--epoc_interval', type=int, default=10,
                        help='Number of epocs between each saving of the weights. Only used with flickr.')
    parser.add_argument('-p', '--print', action='store_true',
                        help='Do not run the training, just print the configuration file. Useful for debug')
    parser.add_argument('--verbose', action='store_true',
                        help='Print more things, useful for debug')

# load parameters
    parser.add_argument('-f', '--force_load', action='store_true',
                        help='Force to load a file even if with differet or missing parameters')
    parser.add_argument('--force_current', action='store_true',
                        help='Force to use the current setting for the parameters')
    parser.add_argument('--start_from', type=str, default='',
                        help='Load a pretrained model from a file. It keeps the current configuration setting. The user should check that the current settings are compatible with the loaded model!')
    parser.add_argument('--convert_transformer', action='store_true',
                        help='Convert the loaded parameters (using start_from) from tensor_add_conv 3x3 to transformer')
    parser.add_argument('--restart', action='store_true',
                        help='Restart from the last iteration of the current filename')
    parser.add_argument('--load_voc_from', type=str, default='',
                        help='Instead of using the vocabulary from the training data it loads it from another training file')

# model parameters
    parser.add_argument('-r', '--conv_reduced', type=int, default=2,
                        help='Reduces the size of the convolutional features from 14x14 to 7x7 to get some speed-up 1: reduces with fix grid 2:reduces with random samples within the grid')
    parser.add_argument('--im_size', type=int, default=400,
                        help='The size of the image used for computing proposals')
    parser.add_argument('-s', '--small_dataset', action='store_true',
                        help='Uses a dataset of only 1k samples. Useful for debug.')
    parser.add_argument('--tensor_cond_word', action='store_true',
                        help='Use the tensor feedback conditioned to the previous word instead of marginalizing')
    parser.add_argument('--rectify', action='store_true',
                        help='Use Rectified unit after Tensor in case of mode (2) tensor-reducedw')
    parser.add_argument('--plot_loss', action='store_true',
                        help='Plot the loss in a graph!')
    parser.add_argument('--proposals', type=int, default=0,
                        help='Train using proposals with different configurations. Valid values are 2,3,4.')
    parser.add_argument('--shuffle_boxes', type=int, default=0,
                        help='When using prosals, shuffle the selected boxes')
    parser.add_argument('--train_only_rnn', action='store_true',
                        help='Train only the RNN part of the network!')
    parser.add_argument('--eval_fixed', action='store_true', default=False,
                        help='Use a fixed set of data for evaluating the loss in training and evaluation. Less variance but biased.')
    parser.add_argument('--conv_normalized', type=int,
                        default=2, help='Use Normalized Convolution Features')
    parser.add_argument('--start_normalized', type=int, default=2,
                        help='Use Normalized Features for the starting point')
    #parser.add_argument('--init_cnn_lasagne', type=int, default=0, help='Which cnn to use as initial descriptor (fc7 layer). Default is pretrained Karphaty(0), but for fine tuning is better to use lasagne (1)')

# optimization parameters
    parser.add_argument('--lr', dest='learning_rate', type=float,
                        default=CFG['LR'], help='Set the learning rate.')
    parser.add_argument('--lr_decay_start', type=int, default=100,
                        help='Set at which epoc starting to use learning decay, defautl 100')
    parser.add_argument('--lr_decay_epocs', type=float, default=2,
                        help='Set every how many epocs reduce the leraning rate by a factor 2')
    parser.add_argument('--max_epocs', dest='max_epocs',
                        type=int, default=400, help='Maximum number of epocs')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=CFG['BATCH_SIZE'], help='Set the batch size. Note that a smaller batch size will require a smaller learning rate as well.')
    parser.add_argument('--grad_clip', type=float, default=100.,
                        help='Set the value for clipping the gradient.')
    parser.add_argument('--reg_h', dest='reg_h', type=int,
                        default=CFG['REG_H'], help='Regularize h to have similar norm at each iteration')
    parser.add_argument('--add_validation', action='store_true',
                        help='Add 30K of validation data as additional training data')

# model parameters
    parser.add_argument('--tensor_tied', dest='tensor_tied', type=int, default=1,
                        help='In mode tensor_feedback(4) and tensor_feedback2(5) defines if the two tensors are tied or not. 0:not tied 1:fully tied 2:only rw is tied (not implemented)')
    parser.add_argument('--reducedw_size', dest='reducedw_size', type=int,
                        default=CFG['EMBEDDING_WORDS'], help='Set the size of the reducedw in mode (2) tensor-reducedw')
    parser.add_argument('--softmax', type=str, default='accurate',
                        help='Using cuDNN you can choose either fast or accurate softmax')
    parser.add_argument('--max_sent', dest='max_sent', type=int,
                        default=CFG['SEQUENCE_LENGTH'], help='Maximum number of words in a sentence')
    parser.add_argument('--new_save', type=int, default=1,
                        help='New format of assigning names to the layers of the models')
    parser.add_argument('--use_flip', type=int, default=0,
                        help='Sample from also flipped images')
    parser.add_argument('--feedback', dest='feedback', type=int, default=2,
                        help='Use different kind of feedback for mode(5) 0:No feedback (as mode 1 but slower), 1:Geometrical feedback (as mode 4) 2: Visual feedback (default) 3: Combination of 1 and 2, 4: Image with regions weighted by their prob.')
    parser.add_argument('--rnn_dropout', type=float,
                        default=0.5, help='Use RNN dropout (default 0.5)')
    parser.add_argument('--it_size', type=int, default=1000,
                        help='Number of samples to process during an iteration')
    parser.add_argument('--dropout_regions', type=int, default=0, help='Use a dropout also for the region descriptors before feeding the tensor. 0: No dropour 1: Dropout same for each timestep and each sentence 2: Dropout same for each timestep but different for each sentence 3: Dropout different for each timestep and for each sentence')
    parser.add_argument('--state_size', type=int,
                        default=CFG['EMBEDDING_SIZE'], help='Size of the state of the RNN')
    parser.add_argument('--repeated_words', type=int, default=10,
                        help='Minimum number of repetition of a words in the training data to add it in the word dictionary.')
    parser.add_argument('--num_proposals', type=int,
                        default=50, help='Number of used proposals')
    parser.add_argument('--clean_masks', type=int, default=1,
                        help='Clean the masks associated to sentences. To remove the bug is should be set to 1')
    parser.add_argument('--save_partial', type=int, default=-1,
                        help='Save partial results every N/1000 batches. Useful in besteffort mode. A good value should be between 5 and 10 depending on the speed of the machine')

    parser.add_argument('--cnn_model', type=str, default='vgg',
                        help='Chose CNN among "vgg" and "resnet". Default is VGG')
    parser.add_argument('--resnet_layer', type=str, default='pool5',
                        help='Use either "prob" or "pool5" layers. Defalult is pool5')
    parser.add_argument('--review_attention', action='store_true',
                        help='Use a mechanism to review the attention mechanism')
    parser.add_argument('--density_tempering', action='store_true',
                        help='Use a mechanism to sharpen or blunt the density distribution.')
    parser.add_argument('--tensor_add_conv', action='store_true',
                        help='Add an additional learnable 3x3 convolutional layer before the tensor')
    parser.add_argument('--tensor_add_conv_nolearn', action='store_true',
                        help='Not learn the added convolutional layer. Later try to learn it with the fine tuning of the CNN.')

    # Options used for image feedback. Highres will upsamle alphas.
    # drop_imfeedback will add some dropout to the convnet used for feedback.
    parser.add_argument('--imgfeedback_mechanism', type=str, default='simple',
                        help='Choose the mechanism to generate the density tempering parameter. Choices: "simple", "highres"')
    parser.add_argument('--drop_imfeedback', type=int, default=0,
                        help=' If set to 1, drops a layer of the image feedback mechanism.')

    # transformer options
    parser.add_argument('--trans_multiple_boxes', type=int, default=1,
                        help='If set to 1 it uses spatial transformer with multiple boxes')
    parser.add_argument('--trans_use_state', type=int, default=0,
                        help='If set to 1 it uses the state vector for the generation of the multiple boxes')
    parser.add_argument('--trans_locnet', type=int, default=0,
                        help='0: loc net first layer uses relu 1: loc net first layer does not use relu, to avoid rotations of the bboxes')
    parser.add_argument('--trans_nolearn', action='store_true',
                        help='Does not learn the localization network, so that resutls should be comparable to the standard tensor. It is a debug option')
    parser.add_argument('--trans_dense_nolearn', action='store_true',
                        help='Does not learn the dense layer after the sampling of the box. It seems to facilitate the learning. To test!!! Try the same also for proposals V3')
    parser.add_argument('--trans_slowlearn', action='store_true',
                        help='Learn the tranformer network slowly than the rest')
    parser.add_argument('--trans_norot', action='store_true',
                        help='The transormer network learns only translation and zoom, not rotation, so that the output are real bboxes')
    parser.add_argument('--trans_stride', type=int, default=1,
                        help='Stride of the initial windows for the spatial transoformer. Default 1')
    parser.add_argument('--trans_use_pretrained', action='store_true',
                        help='Use layers from vgg also for the spatial transformer, so that there are no issues with training other layers')
    parser.add_argument('--trans_zeropad', type=int, default=0,
                        help='Use zero padding for spatial transform, as in convolution')
    parser.add_argument('--trans_reglearn', type=float,
                        default=0., help='Regularize the localization network')
    parser.add_argument('--trans_locnet_dropout', type=float, default=0.,
                        help='Amount of Dropout in the localization network')
    parser.add_argument('--trans_regions_dropout', type=float,
                        default=0.5, help='Amount of Dropout in the localized regions')
    parser.add_argument('--trans_locnet_init', type=float, default=0.1,
                        help='Initializes the localization net with the given variance')
    parser.add_argument('--trans_zoom', type=float, default=1.0,
                        help='Set the zoom of the initial windows')
    parser.add_argument('--trans_compress', type=int, default=0,
                        help='Compress the memory for spatial transformer. Sometimes it has problems. If you find any problem disable it!')
    parser.add_argument('--trans_add_big_proposals', type=int, default=0,
                        help='Add big proposals samples with a stride of 3, so with standard (14,14) map you get 25 additional proposals')
    parser.add_argument('--trans_feedback', type=int, default=0,
                        help='Set to one to use feedback with spatial transformer')

    parser.add_argument('--cnn_slowlearn', type=int, default=0,
                        help='Use a slower learning rate for the CNN')
    parser.add_argument('--set_seed', type=int, default=0,
                        help='Adds the parameter to the seed')

    # Dissected tensor to test WR and RS separately.
    parser.add_argument('--dissect', type=str, default='No',
                        help='Omit one term of the tensor. "wr" will OMMIT rs and "rs" will OMMIT wr ')

    # Try fix
    parser.add_argument('--skip_zero', action='store_true',
                        help='Dont save partial at the beginning. Might not be valid.')
    # }}}

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    loss_train = []
    loss_val = []
    loss_tr = []
    random.seed(3)

    # Turn to config {{{ 1
    CFG['FILE_NAME'] = params['filename']

    mode_nb_to_name = {0: 'normal', 1: 'tensor', 2: 'tensor-reducedw', 3: 'tensor-removedWrw', 4: 'tensor-feedback',
                       5: 'tensor-feedback2', 6: 'tensor-feedbackH', 7: 'lrcn', 8: 'transformer'}
    CFG['MODE'] = mode_nb_to_name[params['mode']]

    #dropout_value = 0.0
    if params['use_flip']:
        CFG['USE_FLIP'] = 'sample'
    else:
        CFG['USE_FLIP'] = None

    cfg_from_params = ['DATASET', 'CNN_SLOWLEARN', 'LOAD_VOC_FROM', 'TRANS_ADD_BIG_PROPOSALS', 'TRANS_COMPRESS',
                       'TRANS_ZOOM', 'TRANS_REGIONS_DROPOUT', 'TRANS_LOCNET_INIT', 'TRANS_LOCNET_DROPOUT', 'TRANS_ZEROPAD',
                       'TRANS_USE_PRETRAINED', 'TRANS_STRIDE', 'TRANS_FEEDBACK', 'CONVERT_TRANSFORMER', 'SAVE_PARTIAL',
                       'TENSOR_ADD_CONV_NOLEARN', 'TRANS_DENSE_NOLEARN', 'TENSOR_ADD_CONV', 'TRANS_LOCNET', 'TRANS_NOLEARN',
                       'TRANS_SLOWLEARN', 'TRANS_REGLEARN', 'TRANS_NOROT', 'TRANS_MULTIPLE_BOXES', 'TRANS_USE_STATE', 'CLEAN_MASKS',
                       'REVIEW_ATTENTION', 'DENSITY_TEMPERING', 'DROPOUT_REGIONS', 'SOFTMAX', 'GRAD_CLIP', 'PROPOSALS',
                       'IMGFEEDBACK_MECHANISM', 'DROP_IMFEEDBACK', 'LR_DECAY_START', 'LR_DECAY_EPOCS', 'EVAL_FIXED', 'TRAIN_ONLY_RNN',
                       'RNN_DROPOUT', 'CONV_NORMALIZED', 'START_NORMALIZED', 'START_FROM', 'NEW_SAVE', 'FEEDBACK', 'TENSOR_COND_WORD',
                       'TENSOR_TIED', 'REG_H', 'BATCH_SIZE', 'RESTART', 'CONV_REDUCED', 'IM_SIZE', 'REPEATED_WORDS', 'ADD_VALIDATION',
                       'DISSECT', 'SET_SEED']
    CFG.update({key: params[key.lower()] for key in cfg_from_params})

    # TODO check this parameter.
    CFG['TRAIN_ONLY_CNN'] = False
    CFG['CNN_FINE_TUNE'] = True
    CFG['SEQUENCE_LENGTH'] = params['max_sent']
    CFG['EPOCS'] = params['max_epocs']
    CFG['EMBEDDING_WORDS'] = params['reducedw_size']
    CFG['LR'] = params['learning_rate']
    CFG['EMBEDDING_SIZE'] = params['state_size']
    if CFG['TRANS_STRIDE'] > 1:
        CFG['CONV_REDUCED'] = 0
    if CFG['CONV_REDUCED']:
        CFG['NUM_REGIONS'] = 7 * 7
    else:
        CFG['NUM_REGIONS'] = 14 * 14
    if CFG['TRAIN_ONLY_CNN']:
        assert(CFG['CNN_FINE_TUNE'])
    if CFG['PROPOSALS']:
        CFG['REGION_SIZE'] = 512
        CFG['NUM_REGIONS'] = params['num_proposals']
    CFG['CNN_MODEL'] = params['cnn_model']
    # If we use resnet, penultimate layer has 1000 units.
    if CFG['CNN_MODEL'] == 'resnet':
        CFG['RESNET_LAYER'] = params['resnet_layer']
        print(bcolors.FAIL +
              "WARNING: over-riding CFG['REGION_SIZE'] constant to 1024." + bcolors.ENDC)
        CFG['REGION_SIZE'] = 1024
        if CFG['RESNET_LAYER'] == 'prob':
            CFG['CNN_FEATURE_SIZE'] = 1000
            print("Chose prob layer")  # TODO del
        elif CFG['RESNET_LAYER'] == 'pool5':
            CFG['CNN_FEATURE_SIZE'] = 2048
            print("Chose pool5 layer")  # TODO del
        else:
            print ("Unknown or unsupported layer choice for resnet")
    # }}}

    print CFG

    #softmax = lasagne.nonlinearities.softmax
    mysoftmax = theano.sandbox.cuda.dnn.GpuDnnSoftmax(
        'bc01', CFG['SOFTMAX'], 'channel')

    def softmax(x):
        shape = T.shape(x)
        return mysoftmax(x.reshape((shape[0], shape[1], 1, 1)))

    if params['dataset'] == 'coco':
        print("Loading Coco Data...")
    elif params['dataset'] == 'flickr':
        print("Loading Flickr30k Data...")

    import time
    t = time.time()

    import PrepareData

    if CFG['DATASET'] == "coco":
        dbtrain, dbval = load_coco(
            small=params['small_dataset'], add_validation=CFG['ADD_VALIDATION'], check_overlap=False)
    elif CFG['DATASET'] == "flickr":
        dbtrain, dbval, _ = load_flickr(small=params['small_dataset'])
    else:
        print("Unrecognized dataset :(")
        import pdb
        pdb.set_trace()

    # Count words occuring at least 5 times and construct mapping int <-> word
    allwords = Counter()
    for key, item in dbtrain.items():
        if CFG['DATASET'] == 'coco':
            sentences = item['caption'][0][0]
        elif CFG['DATASET'] == 'flickr':
            sentences = item['caption']
        else:
            print("Unknown dataset choice")
            import pdb
            pdb.set_trace()

        for sentence in sentences:
            allwords.update(PrepareData.tokenize(sentence))

    vocab = [k for k, v in allwords.items() if v >= CFG['REPEATED_WORDS']]
    vocab.insert(0, '#START#')
    vocab.append('#END#')
    vocab.append('#NAW#')  # not a word
    print('Vocabulary size: {} extracted from at least {} repeated words'.format(
        len(vocab), CFG['REPEATED_WORDS']))

    word_to_index = {w: i for i, w in enumerate(vocab)}
    index_to_word = {i: w for i, w in enumerate(vocab)}

    if CFG['LOAD_VOC_FROM'] != '':
        dvoc = pickle.load(open(CFG['START_FROM']))
        vocab = dvoc['vocab']
        word_to_index = dvoc['word_to_index']
        index_to_word = dvoc['index_to_word']
        print('New Vocabulary size: {} extracted from at least {} repeated words'.format(
            len(vocab), CFG['REPEATED_WORDS']))

    l_input_sentence = InputLayer(
        (CFG['BATCH_SIZE'] * 5, CFG['SEQUENCE_LENGTH'] - 1), name='l_input_sentence')
    l_input_img = InputLayer(
        (CFG['BATCH_SIZE'], 3, 224, 224), name='cnn_input')

    if CFG['TRANS_USE_PRETRAINED']:
        if CFG['CNN_MODEL'] == 'resnet':
            l_dropout_input, l_conv_input, l_conv_input2 = build_seq_input_net(
                CFG, vocab, l_input_sentence, l_input_img)
        else:
            l_dropout_input, l_conv_input, l_conv_input2 = build_seq_input_net(
                CFG, vocab, l_input_sentence, l_input_img)
    else:
        l_dropout_input, l_conv_input = build_seq_input_net(
            CFG, vocab, l_input_sentence, l_input_img)

    if CFG['MODE'] == 'normal':
        # {{{ 1
        print(bcolors.OKGREEN + "Normal mode" + bcolors.ENDC)
        # define a cell

        l_cell_input = InputLayer(
            (CFG['BATCH_SIZE'] * 5, CFG['EMBEDDING_SIZE']), name='l_cell_input')
        from agentnet.memory import GRUMemoryLayer
        l_prev_gru = InputLayer(
            (CFG['BATCH_SIZE'] * 5, CFG['EMBEDDING_SIZE']), name="l_prev_gru")
        l_gru = GRUMemoryLayer(CFG['EMBEDDING_SIZE'],
                               l_cell_input, l_prev_gru, name='l_gru')

        l_dropout_output = lasagne.layers.DropoutLayer(
            l_gru, p=CFG['RNN_DROPOUT'], name='l_dropout_output')

        l_decoder = DenseLayer(l_dropout_output, num_units=len(
            vocab), nonlinearity=softmax, name='l_decoder')

        from collections import OrderedDict
        memory_dict = OrderedDict([
            (l_gru, l_prev_gru),
        ])

        from agentnet.agent import Recurrence
        l_rec = Recurrence(

            # we use out previously defined dictionary to update recurrent network parameters
            state_variables=memory_dict,

            # we feed in reference sequence into "prev letter" input layer, tick by tick along the axis=1.
            input_sequences={l_cell_input: l_dropout_input},

            # we track agent would-be actions and probabilities
            tracked_outputs=l_decoder,

            n_steps=CFG['SEQUENCE_LENGTH'],

            # finally, we define an optional batch size param
            #(if omitted, it will be inferred from inputs or initial value providers if there are any)
            batch_size=CFG['BATCH_SIZE'] * 5,
        )

        l_gru_states, l_out = l_rec.get_sequence_layers()
        #}}}

    elif CFG['MODE'] == 'tensor':
        # {{{ 1
        print(bcolors.OKGREEN + "Tensor mode." + bcolors.ENDC)
        # define a cell
        l_cell_input = InputLayer(
            (CFG['BATCH_SIZE'] * 5, CFG['EMBEDDING_SIZE']), name='l_cell_input')
        from agentnet.memory import GRUMemoryLayer
        l_prev_gru = InputLayer(
            (CFG['BATCH_SIZE'] * 5, CFG['EMBEDDING_SIZE']), name="l_prev_gru")
        l_gru = GRUMemoryLayer(CFG['EMBEDDING_SIZE'],
                               l_cell_input, l_prev_gru, name='l_gru')
        l_dropout_output = lasagne.layers.DropoutLayer(
            l_gru, p=CFG['RNN_DROPOUT'], name='l_dropout_output')

        from TProd3 import TensorProdFactLayer, SubsampleLayer, Multiplex
        if CFG['CNN_FINE_TUNE']:
            # use images at different resolution but without fully connected layers
            if CFG['PROPOSALS'] == 3:
                vgg16_det = CNN.build_model_RCNN(
                    CFG['NUM_REGIONS'], im_size=CFG['IM_SIZE'] * 1.5, pool_dims=3, dropout_value=CFG['RNN_DROPOUT'])
                print "Loading pretrained VGG16 parameters for detection"
                model_param_values = pickle.load(open('vgg16.pkl'))[
                    'param values']
                lasagne.layers.set_all_param_values(
                    vgg16_det['conv5_3'], model_param_values[:-6])
                l_input_img2 = vgg16_det['input']
                l_boxes = vgg16_det['boxes']
                l_input_regions = vgg16_det['reshape']
                if CFG['CONV_NORMALIZED'] == 1:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                if CFG['CONV_NORMALIZED'] == 2:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                else:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X * 0.01, output_shape='auto')
                # added because without it overfits fast, but it learns also fast
                l_input_regions = dropout(l_input_regions, p=0.5)
                l_cnn_proposals = DenseLayer(
                    l_input_regions, num_units=CFG['REGION_SIZE'], name='l_cnn_proposals')
                l_input_regions = ReshapeLayer(
                    l_cnn_proposals, (CFG['BATCH_SIZE'], CFG['NUM_REGIONS'], CFG['REGION_SIZE']))
                l_input_regions = lasagne.layers.DimshuffleLayer(
                    l_input_regions, (0, 2, 1))
                l_input_regions = lasagne.layers.ExpressionLayer(
                    l_input_regions, lambda X: theano.tensor.extra_ops.repeat(X, 5, axis=0), output_shape='auto')
                l_input_regions = ReshapeLayer(
                    l_input_regions, (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE'], CFG['NUM_REGIONS']))

            # use the second branch with proposals without any piramid with max-polling
            elif CFG['PROPOSALS'] == 4:
                vgg16_det = CNN.build_model_RCNN(CFG['NUM_REGIONS'], im_size=int(
                    CFG['IM_SIZE'] * 1.5), pool_dims=1, dropout_value=CFG['RNN_DROPOUT'])
                print "Loading pretrained VGG16 parameters for detection"
                model_param_values = pickle.load(open('vgg16.pkl'))[
                    'param values']
                lasagne.layers.set_all_param_values(
                    vgg16_det['conv5_3'], model_param_values[:-6])
                l_input_img2 = vgg16_det['input']
                l_boxes = vgg16_det['boxes']
                l_input_regions = vgg16_det['crop']
                l_input_regions = ReshapeLayer(
                    l_input_regions, (CFG['BATCH_SIZE'] * CFG['NUM_REGIONS'], CFG['REGION_SIZE']))

                if CFG['CONV_NORMALIZED'] == 1:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                if CFG['CONV_NORMALIZED'] == 2:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                else:

                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X * 0.01, output_shape='auto')

                l_input_regions = ReshapeLayer(
                    l_input_regions, (CFG['BATCH_SIZE'], CFG['NUM_REGIONS'], CFG['REGION_SIZE']))
                l_input_regions = lasagne.layers.DimshuffleLayer(
                    l_input_regions, (0, 2, 1))
                l_input_regions = lasagne.layers.ExpressionLayer(
                    l_input_regions, lambda X: theano.tensor.extra_ops.repeat(X, 5, axis=0), output_shape='auto')
                l_input_regions = ReshapeLayer(
                    l_input_regions, (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE'], CFG['NUM_REGIONS']))

            # use the second branch (vgg16_det) with grid porposals from conv5_3
            elif CFG['PROPOSALS'] == 5:
                vgg16_det = CNN.build_model_RCNN(CFG['NUM_REGIONS'], im_size=int(
                    CFG['IM_SIZE'] * 1.5), pool_dims=1, dropout_value=CFG['RNN_DROPOUT'])
                print "Loading pretrained VGG16 parameters for detection"
                model_param_values = pickle.load(open('vgg16.pkl'))[
                    'param values']
                lasagne.layers.set_all_param_values(
                    vgg16_det['conv5_3'], model_param_values[:-6])
                l_input_img2 = vgg16_det['input']
                l_input_regions = vgg16_det['conv5_3']
                if CFG['CONV_REDUCED'] == 1:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X[:, :, ::2, ::2], output_shape='auto')
                elif CFG['CONV_REDUCED'] == 2:
                    l_input_regions = SubsampleLayer(l_input_regions, stride=2)

                if CFG['CONV_NORMALIZED'] == 1:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                if CFG['CONV_NORMALIZED'] == 2:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                else:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X * 0.01, output_shape='auto')
                l_input_regions = lasagne.layers.ExpressionLayer(
                    l_input_regions, lambda X: theano.tensor.extra_ops.repeat(X, 5, axis=0), output_shape='auto')
                l_input_regions = ReshapeLayer(
                    l_input_regions, (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE'], CFG['NUM_REGIONS']))

            else:  # use conv5_3 from the normal vgg16 branch
                l_input_regions = l_conv_input
                if CFG['CONV_REDUCED'] == 1:
                    # added a scaling factor of 100 to avoid exploding gradients
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X[:, :, ::2, ::2], output_shape='auto')
                elif CFG['CONV_REDUCED'] == 2:
                    l_input_regions = SubsampleLayer(l_input_regions, stride=2)
                # else:
                if CFG['CONV_NORMALIZED'] == 1:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                elif CFG['CONV_NORMALIZED'] == 2:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                else:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X * 0.01, output_shape='auto')
                if CFG['DROPOUT_REGIONS'] == 1:
                    l_input_regions = dropout(l_input_regions, p=0.5)
                l_input_regions = lasagne.layers.ExpressionLayer(
                    l_input_regions, lambda X: theano.tensor.extra_ops.repeat(X, 5, axis=0), output_shape='auto')
                if CFG['DROPOUT_REGIONS'] == 2:
                    l_input_regions = dropout(l_input_regions, p=0.5)
                if CFG['TENSOR_ADD_CONV']:
                    l_input_regions = ConvLayer(l_input_regions, num_filters=CFG['REGION_SIZE'], filter_size=(
                        3, 3), pad='same', name='l_add_con')
                    if CFG['TENSOR_ADD_CONV_NOLEARN']:
                        set_all_layers_tags(l_input_regions, treat_as_input=[
                                            l_conv_input], trainable=False)

                l_input_regions = ReshapeLayer(
                    l_input_regions, (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE'], CFG['NUM_REGIONS']))
        else:
            l_input_regions = InputLayer(
                (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regions')

        l_input_regions_cell = InputLayer(
            (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regions_cell')
        if CFG['DROPOUT_REGIONS'] == 3:
            l_input_regions_cell = dropout(l_input_regions_cell, p=0.5)
        # reshape RNN output to be (batch, seq=1, embedding)
        l_shp1 = ReshapeLayer(
            l_dropout_output, ([0], 1, [1]), name='l_shp1')
        if CFG['DISSECT'] == 'wr':
            l_tensor = TensorProdFactLayer((l_shp1, l_input_regions_cell), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'],
                                           dim_w=len(vocab), nonlinearity=softmax, name='l_tensor', W_hr='skip', b_hr='skip')
        elif CFG['DISSECT'] == 'rs':
            l_tensor = TensorProdFactLayer((l_shp1, l_input_regions_cell), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'],
                                           dim_w=len(vocab), nonlinearity=softmax, name='l_tensor', W_rw='skip', b_rw='skip')
        else:
            l_tensor = TensorProdFactLayer((l_shp1, l_input_regions_cell), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
                vocab), nonlinearity=softmax, name='l_tensor')
        l_word = lasagne.layers.ExpressionLayer(l_tensor, lambda X: X.sum(
            2), output_shape='auto', name='l_word')  # sum over regions
        from collections import OrderedDict
        memory_dict = OrderedDict([(l_gru, l_prev_gru)])

        from agentnet.agent import Recurrence
        l_rec = Recurrence(
            input_nonsequences={l_input_regions_cell: l_input_regions},
            # we use out previously defined dictionary to update recurrent network parameters
            state_variables=memory_dict,
            # we feed in reference sequence into "prev letter" input layer, tick by tick along the axis=1.
            input_sequences={l_cell_input: l_dropout_input},
            # we track agent would-be actions and probabilities
            tracked_outputs=l_word,
            n_steps=CFG['SEQUENCE_LENGTH'],
            # finally, we define an optional batch size param
            #(if omitted, it will be inferred from inputs or initial value providers if there are any)
            batch_size=CFG['BATCH_SIZE'] * 5
        )

        l_gru_states, l_out = l_rec.get_sequence_layers()
        # }}}

    elif CFG['MODE'] == 'transformer':
        #{{{ 1
        print(bcolors.OKGREEN + "Transformer mode." + bcolors.ENDC)
        from TProd3 import TensorProdFactLayer, WeightedSumLayer, SubsampleLayer

        # define a cell
        l_cell_input = InputLayer(
            (CFG['BATCH_SIZE'] * 5, CFG['EMBEDDING_SIZE']), name='l_cell_input')
        from agentnet.memory import GRUMemoryLayer
        l_prev_gru = InputLayer(
            (CFG['BATCH_SIZE'] * 5, CFG['EMBEDDING_SIZE']), name="l_prev_gru")

        if CFG['TRANS_FEEDBACK']:
            l_weighted_region_prev = InputLayer(
                (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE']), name="l_weighted_region_prev")
            if CFG['FEEDBACK'] == 2:
                l_cell_concat = ConcatLayer(
                    [l_cell_input, l_weighted_region_prev], axis=1, name='l_cell_concat')
            else:
                print("Are you sure you don't want to use feedback=2? I think you should. Change your mind, then come to see me again.")
        else:
            l_cell_concat = l_cell_input

        l_gru = GRUMemoryLayer(CFG['EMBEDDING_SIZE'],
                               l_cell_concat, l_prev_gru, name='l_gru')
        l_dropout_output = lasagne.layers.DropoutLayer(
            l_gru, p=CFG['RNN_DROPOUT'], name='l_dropout_output')

        from TProd3 import TensorProdFactLayer, SubsampleLayer, Multiplex
        # if I use stride2, I use conv5_2 as input and conv5_3 for the spatial transformer
        if CFG['TRANS_USE_PRETRAINED']:
            l_input_regions = l_conv_input  # vgg16['conv5_2']
        else:
            l_input_regions = l_conv_input2  # vgg16['conv5_3']
        if CFG['CONV_REDUCED'] == 1:
            # added a scaling factor of 100 to avoid exploding gradients
            l_input_regions = lasagne.layers.ExpressionLayer(
                l_input_regions, lambda X: X[:, :, ::2, ::2], output_shape='auto')
        elif CFG['CONV_REDUCED'] == 2:
            l_input_regions = SubsampleLayer(l_input_regions, stride=2)
        # else:
        if CFG['TRANS_USE_PRETRAINED']:
            l_input_regions = l_input_regions
        else:
            if CFG['CONV_NORMALIZED'] == 1:
                l_input_regions = lasagne.layers.ExpressionLayer(
                    l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
            elif CFG['CONV_NORMALIZED'] == 2:
                l_input_regions = lasagne.layers.ExpressionLayer(
                    l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
            else:
                l_input_regions = lasagne.layers.ExpressionLayer(
                    l_input_regions, lambda X: X * 0.01, output_shape='auto')

        if CFG['DROPOUT_REGIONS'] == 1:
            l_input_regions = dropout(l_input_regions, p=0.5)
        expand = 5
        if CFG['TRANS_COMPRESS']:
            expand = 1
        if expand == 5:
            l_input_regions = lasagne.layers.ExpressionLayer(
                l_input_regions, lambda X: theano.tensor.extra_ops.repeat(X, 5, axis=0), output_shape='auto')
        if CFG['DROPOUT_REGIONS'] == 2:
            l_input_regions = dropout(l_input_regions, p=0.5)
        if CFG['CONV_REDUCED'] == 0:
            l_input_regions_cell = InputLayer(
                (CFG['BATCH_SIZE'] * expand, CFG['REGION_SIZE'], 14, 14), name='l_input_regions_cell')
        else:
            l_input_regions_cell = InputLayer(
                (CFG['BATCH_SIZE'] * expand, CFG['REGION_SIZE'], 7, 7), name='l_input_regions_cell')
        #b = np.zeros((2, 3), dtype='float32')
        factor = 2.0
        if CFG['TRANS_LOCNET_INIT'] > 0.:
            W = lasagne.init.HeNormal(CFG['TRANS_LOCNET_INIT'])  # 0.01
        else:
            W = lasagne.init.Constant(0.)
        W0 = lasagne.init.Constant(0.)
        b = lasagne.init.Constant(0.)
        if CFG['TRANS_MULTIPLE_BOXES']:
            num_prop, l_tr_loc = build_loc_net(CFG, l_gru, l_input_regions_cell, expand, (
                14, 14), (3, 3), CFG['TRANS_STRIDE'], CFG['TRANS_ZOOM'], W, b, name='')
            # big proposals
            if CFG['TRANS_ADD_BIG_PROPOSALS']:
                num_prop_big, l_tr_loc_big = build_loc_net(CFG, l_gru, l_input_regions_cell, expand, (
                    14, 14), (3, 3), CFG['TRANS_STRIDE'], CFG['TRANS_ZOOM'] * 2, W, b, name='_big')
                l_tr_loc = ConcatLayer((l_tr_loc, l_tr_loc_big), axis=0)
                num_prop += num_prop_big
            # end big proposals
            # translate the pooling region to make it translation invariant
            # it's important to not mess up with the coordinates!!!
            # if expand ==1:
            l_sel_region = MultiTransformerLayer(l_input_regions_cell, l_tr_loc, kernel_size=(
                3, 3), zero_padding=CFG['TRANS_ZEROPAD'])  # 3x3
            if CFG['TRANS_USE_PRETRAINED']:
                if CFG['CNN_MODEL'] == 'vgg':
                    Wvgg = l_conv_input2.W
                    Wvgg = Wvgg.reshape(
                        (CFG['REGION_SIZE'], CFG['REGION_SIZE'] * 3 * 3)).swapaxes(0, 1)
                    bvgg = l_conv_input2.b
                    l_sel_region = DenseLayer(
                        l_sel_region, num_units=CFG['REGION_SIZE'], name='l_sel_region', W=Wvgg, b=bvgg)
                elif CFG['CNN_MODEL'] == 'resnet':
                    print(
                        "Resnet + transformer + feedback is not properly implemented yet. Exiting")
                    sys.exit()
                    Wresnet = l_conv_input2.W
                    # Who doesn't like hard coded constants?
                    Wresnet = Wresnet.reshape(
                        (256, 256 * 3 * 3)).swapaxes(0, 1)
                    bresnet = l_conv_input2.b
                    l_sel_region = DenseLayer(
                        l_sel_region, num_units=256, name='l_sel_region', W=Wresnet, b=bresnet)

                if CFG['CONV_NORMALIZED'] == 1:
                    l_sel_region = lasagne.layers.ExpressionLayer(
                        l_sel_region, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                elif CFG['CONV_NORMALIZED'] == 2:
                    l_sel_region = lasagne.layers.ExpressionLayer(
                        l_sel_region, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                else:
                    l_sel_region = l_sel_region
            else:
                l_sel_region = DenseLayer(
                    l_sel_region, num_units=CFG['REGION_SIZE'], name='l_sel_region')
                # l_sel_region =

            if CFG['TRANS_DENSE_NOLEARN']:
                set_all_layers_tags(l_sel_region, treat_as_input=[
                                    l_input_regions_cell, l_tr_loc], trainable=False, nolearn=True)
            else:
                set_all_layers_tags(l_sel_region, treat_as_input=[
                                    l_input_regions_cell, l_tr_loc], trainable=True, nolearn=True)
            l_sel_region = ReshapeLayer(
                l_sel_region, (CFG['BATCH_SIZE'] * expand, num_prop, CFG['REGION_SIZE']))
            l_sel_region = DimshuffleLayer(l_sel_region, (0, 2, 1))
            l_sel_region = ReshapeLayer(
                l_sel_region, (CFG['BATCH_SIZE'] * expand, CFG['REGION_SIZE'], num_prop))
            if expand == 1:
                l_sel_region = lasagne.layers.ExpressionLayer(
                    l_sel_region, lambda X: theano.tensor.extra_ops.repeat(X, 5, axis=0), output_shape='auto')
        else:
            print("Warning: transform no multibox is flagged as a deprecated option.")
            b[0, 0] = factor
            b[1, 1] = factor
            b = b.flatten()
            l_input_loc = l_gru
            if CFG['TRANS_USE_STATE']:
                l_input_im = ConvLayer(l_input_regions_cell, num_filters=512, filter_size=(
                    3, 3), pad='same', name='l_reduce_im1')
                l_input_im = lasagne.layers.MaxPool2DLayer(l_input_im, (2, 2))
                l_input_im = ConvLayer(l_input_im, num_filters=512, filter_size=(
                    3, 3), pad='same', name='l_reduce_im2')
                l_input_im = lasagne.layers.MaxPool2DLayer(l_input_im, (2, 2))
                l_input_im = ReshapeLayer(
                    l_input_im, (CFG['BATCH_SIZE'] * 5, 512))
                l_input_loc = ConcatLayer((l_gru, l_input_im))
            l_loc1 = DenseLayer(
                l_input_loc, num_units=256, name='l_loc1')
            l_loc2 = DenseLayer(
                l_loc1, num_units=6, W=W, b=b, nonlinearity=None, name='l_loc2')
            l_sel_region = TransformerLayer(
                l_input_regions_cell, l_loc2, downsample_factor=factor)
            l_sel_region = dropout(l_sel_region, p=0.5)
            l_sel_region = DenseLayer(
                l_sel_region, num_units=CFG['REGION_SIZE'], name='l_sel_region')
            #  [Marco]: Don't look here either
            l_sel_region = ReshapeLayer(
                l_sel_region, (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE'], 1))
            l_sel_region = dropout(l_sel_region, p=0.5)

        if CFG['DROPOUT_REGIONS'] == 3:
            l_input_regions_cell2 = dropout(
                l_sel_region, p=CFG['TRANS_REGIONS_DROPOUT'])
            l_shp1 = ReshapeLayer(
                l_dropout_output, ([0], 1, [1]), name='l_shp1')

            if CFG['DENSITY_TEMPERING']:
                l_gamma = DenseLayer(
                    l_shp1, num_units=1, name='l_gamma')
                l_gamma_shp = ReshapeLayer(
                    l_gamma, ([0], [1], 1, 1))
                from TProd3 import TensorTemperatureLayer
                l_tensor = TensorTemperatureLayer((l_shp1, l_input_regions_cell2, l_gamma_shp), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
                    vocab), nonlinearity=lasagne.nonlinearities.softmax, name='l_tensor')
            else:
                l_tensor = TensorProdFactLayer((l_shp1, l_input_regions_cell2), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
                    vocab), nonlinearity=softmax, name='l_tensor')
        else:
            l_shp1 = ReshapeLayer(
                l_dropout_output, ([0], 1, [1]), name='l_shp1')
            if CFG['DENSITY_TEMPERING']:
                l_gamma = DenseLayer(
                    l_shp1, num_units=1, name='l_gamma')
                l_gamma_shp = ReshapeLayer(
                    l_gamma, ([0], [1], 1, 1))
                from TProd3 import TensorTemperatureLayer
                l_tensor = TensorTemperatureLayer((l_shp1, l_sel_region, l_gamma_shp), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
                    vocab), nonlinearity=lasagne.nonlinearities.softmax, name='l_tensor')
            else:
                l_tensor = TensorProdFactLayer((l_shp1, l_sel_region), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
                    vocab), nonlinearity=softmax, name='l_tensor')

        l_word = lasagne.layers.ExpressionLayer(l_tensor, lambda X: X.sum(
            2), output_shape='auto', name='l_word')  # sum over regions

        from collections import OrderedDict

        if CFG['TRANS_FEEDBACK']:
            l_region = lasagne.layers.ExpressionLayer(l_tensor, lambda X: X.sum(
                3), output_shape='auto', name='l_region')  # sum over regions
            l_weighted_region = WeightedSumLayer(
                [l_sel_region, l_region], name='l_weighted_region')
            memory_dict = OrderedDict(
                [(l_gru, l_prev_gru), (l_weighted_region, l_weighted_region_prev)])
        else:
            memory_dict = OrderedDict([(l_gru, l_prev_gru)])

        from agentnet.agent import Recurrence
        l_rec = Recurrence(
            input_nonsequences={l_input_regions_cell: l_input_regions},
            # we use out previously defined dictionary to update recurrent network parameters
            state_variables=memory_dict,
            # we feed in reference sequence into "prev letter" input layer, tick by tick along the axis=1.
            input_sequences={l_cell_input: l_dropout_input},
            # we track agent would-be actions and probabilities
            tracked_outputs=l_word,
            n_steps=CFG['SEQUENCE_LENGTH'],
            # finally, we define an optional batch size param
            #(if omitted, it will be inferred from inputs or initial value providers if there are any)
            batch_size=CFG['BATCH_SIZE'] * 5
        )

        l_gru_states, l_out = l_rec.get_sequence_layers()
        #}}}

    elif CFG['MODE'] == 'tensor-feedback':  # deprecated
        #{{{ 1
        print("Warning: tensor-feedback is flagged as a deprecated option.")
        # define a cell
        l_cell_input = InputLayer(
            (CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), name='l_cell_input')
        from agentnet.memory import GRUMemoryLayer
        l_prev_gru = InputLayer(
            (CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), name="l_prev_gru")
        l_region_feedback = InputLayer(
            (CFG['BATCH_SIZE'], CFG['NUM_REGIONS']), name='l_region_feedback')
        l_cell_concat = ConcatLayer(
            [l_cell_input, l_region_feedback], axis=1, name='l_cell_concat')
        l_gru = GRUMemoryLayer(CFG['EMBEDDING_SIZE'],
                               l_cell_concat, l_prev_gru, name='l_gru')
        l_dropout_output = lasagne.layers.DropoutLayer(
            l_gru, p=CFG['RNN_DROPOUT'], name='l_dropout_output')

        from TProd3 import TensorProdFactLayer
        l_input_regions = InputLayer(
            (CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regions')

        l_input_regions_cell = InputLayer(
            (CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regions_cell')
        # reshape RNN output to be (batch, seq=1, embedding)
        l_shp1 = ReshapeLayer(
            l_dropout_output, ([0], 1, [1]), name='l_shp1')
        l_tensor = TensorProdFactLayer((l_shp1, l_input_regions_cell), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
            vocab), nonlinearity=softmax, name='l_tensor')
        l_word = lasagne.layers.ExpressionLayer(l_tensor, lambda X: X.sum(
            2), output_shape='auto', name='l_word')  # sum over
        l_region = lasagne.layers.ExpressionLayer(l_tensor, lambda X: X.sum(
            3), output_shape='auto', name='l_region')  # sum over regions
        l_region = ReshapeLayer(
            l_region, ([0], [2]), name='l_region')
        from collections import OrderedDict
        memory_dict = OrderedDict(
            [(l_gru, l_prev_gru), (l_region, l_region_feedback)])

        from agentnet.agent import Recurrence
        l_rec = Recurrence(
            input_nonsequences={l_input_regions_cell: l_input_regions},
            # we use out previously defined dictionary to update recurrent network parameters
            state_variables=memory_dict,
            # we feed in reference sequence into "prev letter" input layer, tick by tick along the axis=1.
            input_sequences={l_cell_input: l_dropout_input},
            # we track agent would-be actions and probabilities
            tracked_outputs=l_word,
            n_steps=CFG['SEQUENCE_LENGTH'],
            # finally, we define an optional batch size param
            #(if omitted, it will be inferred from inputs or initial value providers if there are any)
            batch_size=CFG['BATCH_SIZE'],
        )

        l_gru_states, l_out = l_rec.get_sequence_layers()
        #}}}

    elif CFG['MODE'] == 'tensor-feedback2':
        #{{{ 1

        from TProd3 import TensorProdFactLayer, WeightedSumLayer, SubsampleLayer

        l_cell_input = InputLayer(
            (CFG['BATCH_SIZE'] * 5, CFG['EMBEDDING_SIZE']), name='l_cell_input')
        if CFG['CNN_FINE_TUNE']:
            if CFG['PROPOSALS'] in [1, 2, 5]:
                print("Proposals methods 1,2,5 should are not implemented with feedback.")
                sys.exit()
            elif CFG['PROPOSALS'] == 3:
                # {{{2
                # use images at different resolution but without fully connected layers
                if CFG['CNN_MODEL'] == 'vgg':
                    vgg16_det = CNN.build_model_RCNN(
                        CFG['NUM_REGIONS'], im_size=CFG['IM_SIZE'] * 1.5, pool_dims=3, dropout_value=CFG['RNN_DROPOUT'])
                    print "Loading pretrained VGG16 parameters for detection"
                    model_param_values = pickle.load(open('vgg16.pkl'))['param values']
                    lasagne.layers.set_all_param_values(
                        vgg16_det['conv5_3'], model_param_values[:-6])
                    l_input_img2 = vgg16_det['input']
                    l_boxes = vgg16_det['boxes']
                    l_input_regions = vgg16_det['reshape']
                else:
                    resnet50_det = resnet_CNN.build_model_RCNN(
                        CFG['NUM_REGIONS'], im_size=CFG['IM_SIZE'] * 1.5, pool_dims=3, dropout_value=CFG['RNN_DROPOUT'])
                    print "Loading pretrained resnet50 parameters for detection"
                    # You can use this format to store other things for best effort
                    model_param_values = pickle.load(open('resnet50.pkl'))[
                        'param values']

                    from save_layers import add_names_layers_and_params
                    add_names_layers_and_params(resnet50_det)
                    #lasagne.layers.set_all_param_values(resnet50['prob'], model_param_values)
                    set_param_dict(
                        resnet50_det['pool5'], model_param_values, prefix='', show_layers=False, relax=False)
                    l_input_img2 = resnet50_det['input']
                    l_boxes = resnet50_det['boxes']
                    l_input_regions = resnet50_det['reshape']

                if CFG['CONV_NORMALIZED'] == 1:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                if CFG['CONV_NORMALIZED'] == 2:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                else:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X * 0.01, output_shape='auto')
                # added because without it overfits fast, but it learns also fast
                l_input_regions = dropout(l_input_regions, p=0.5)
                l_cnn_proposals = DenseLayer(
                    l_input_regions, num_units=CFG['REGION_SIZE'], name='l_cnn_proposals')
                l_input_regions = ReshapeLayer(
                    l_cnn_proposals, (CFG['BATCH_SIZE'], CFG['NUM_REGIONS'], CFG['REGION_SIZE']))
                l_input_regions = lasagne.layers.DimshuffleLayer(
                    l_input_regions, (0, 2, 1))
                l_input_regions = lasagne.layers.ExpressionLayer(
                    l_input_regions, lambda X: theano.tensor.extra_ops.repeat(X, 5, axis=0), output_shape='auto')
                l_input_regions = ReshapeLayer(
                    l_input_regions, (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE'], CFG['NUM_REGIONS']))
                # }}}

            # use the second branch with proposals without any piramid with max-polling
            elif CFG['PROPOSALS'] == 4:
                # {{{ 2
                if CFG['CNN_MODEL'] == 'vgg':
                    vgg16_det = CNN.build_model_RCNN(CFG['NUM_REGIONS'], im_size=int(
                        CFG['IM_SIZE'] * 1.5), pool_dims=1, dropout_value=CFG['RNN_DROPOUT'])
                    print "Loading pretrained VGG16 parameters for detection"
                    model_param_values = pickle.load(open('vgg16.pkl'))[
                        'param values']
                    lasagne.layers.set_all_param_values(
                        vgg16_det['conv5_3'], model_param_values[:-6])
                    l_input_img2 = vgg16_det['input']
                    l_boxes = vgg16_det['boxes']
                    l_input_regions = vgg16_det['crop']
                    l_input_regions = ReshapeLayer(
                        l_input_regions, (CFG['BATCH_SIZE'] * CFG['NUM_REGIONS'], CFG['REGION_SIZE']))
                else:
                    resnet50_det = resnet_CNN.build_model_RCNN(
                        CFG['NUM_REGIONS'], im_size=CFG['IM_SIZE'] * 1.5, pool_dims=1, dropout_value=CFG['RNN_DROPOUT'])
                    print "Loading pretrained resnet50 parameters for detection"
                    # You can use this format to store other things for best effort
                    model_param_values = pickle.load(open('resnet50.pkl'))[
                        'param values']
                    from save_layers import add_names_layers_and_params
                    add_names_layers_and_params(resnet50_det)
                    set_param_dict(
                        resnet50_det['pool5'], model_param_values, prefix='', show_layers=False, relax=False)
                    l_input_img2 = resnet50_det['input']
                    l_boxes = resnet50_det['boxes']
                    l_input_regions = resnet50_det['crop']
                    l_input_regions = ReshapeLayer(
                        l_input_regions, (CFG['BATCH_SIZE'] * CFG['NUM_REGIONS'], CFG['REGION_SIZE']))

                if CFG['CONV_NORMALIZED'] == 1:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                if CFG['CONV_NORMALIZED'] == 2:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                else:

                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X * 0.01, output_shape='auto')

                l_input_regions = ReshapeLayer(
                    l_input_regions, (CFG['BATCH_SIZE'], CFG['NUM_REGIONS'], CFG['REGION_SIZE']))
                l_input_regions = lasagne.layers.DimshuffleLayer(
                    l_input_regions, (0, 2, 1))
                l_input_regions = lasagne.layers.ExpressionLayer(
                    l_input_regions, lambda X: theano.tensor.extra_ops.repeat(X, 5, axis=0), output_shape='auto')
                l_input_regions = ReshapeLayer(
                    l_input_regions, (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE'], CFG['NUM_REGIONS']))
                #}}}

            else:  # use conv5_3 from the normal vgg16 branch
                l_input_regions = l_conv_input  # vgg16['conv5_3']
                if CFG['CONV_REDUCED'] == 1:
                    # added a scaling factor of 100 to avoid exploding gradients
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X[:, :, ::2, ::2], output_shape='auto')
                elif CFG['CONV_REDUCED'] == 2:
                    l_input_regions = SubsampleLayer(l_input_regions, stride=2)
                # else:
                if CFG['CONV_NORMALIZED'] == 1:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                elif CFG['CONV_NORMALIZED'] == 2:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                else:
                    l_input_regions = lasagne.layers.ExpressionLayer(
                        l_input_regions, lambda X: X * 0.01, output_shape='auto')
                if CFG['DROPOUT_REGIONS'] == 1:
                    l_input_regions = dropout(l_input_regions, p=0.5)
                l_input_regions = lasagne.layers.ExpressionLayer(
                    l_input_regions, lambda X: theano.tensor.extra_ops.repeat(X, 5, axis=0), output_shape='auto')
                if CFG['DROPOUT_REGIONS'] == 2:
                    l_input_regions = dropout(l_input_regions, p=0.5)
                if CFG['TENSOR_ADD_CONV']:
                    l_input_regions = ConvLayer(l_input_regions, num_filters=CFG['REGION_SIZE'], filter_size=(
                        3, 3), pad='same', name='l_add_con')
                    if CFG['TENSOR_ADD_CONV_NOLEARN']:
                        set_all_layers_tags(l_input_regions, treat_as_input=[
                                            l_conv_input], trainable=False)

                l_input_regions = ReshapeLayer(
                    l_input_regions, (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE'], CFG['NUM_REGIONS']))
        l_input_regions_cell = InputLayer(
            (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regions_cell')
        l_input_regions_cell_highres = InputLayer(
            (CFG['BATCH_SIZE'] * 5, CFG['REGION_SIZE'], 14 * 14), name='l_input_regions_cell_highres')

        from agentnet.memory import GRUMemoryLayer
        l_prev_gru = InputLayer(
            (CFG['BATCH_SIZE'] * 5, CFG['EMBEDDING_SIZE']), name="l_prev_gru")
        if CFG['TENSOR_TIED']:
            l_region_feedback = InputLayer(
                (CFG['BATCH_SIZE'] * 5, CFG['NUM_REGIONS']), name='l_region_feedback')
            l_region_feedback_shape = ReshapeLayer(
                l_region_feedback, ([0], 1, [1]), name='l_region_feedback_shape')
        else:
            l_shp2 = lasagne.layers.DropoutLayer(
                l_prev_gru, p=CFG['RNN_DROPOUT'], name='l_shp2')
            l_shp2 = ReshapeLayer(l_shp2, ([0], 1, [1]))

            if CFG['DENSITY_TEMPERING']:
                gamma = DenseLayer(l_shp2, num_units=1)
                l_tensor2_unnormalized = TensorProdFactLayer((l_shp2, l_input_regions_cell),
                                                             dim_h=CFG['EMBEDDING_SIZE'],
                                                             dim_r=CFG['REGION_SIZE'],
                                                             dim_w=len(vocab), nonlinearity=linear,
                                                             name='l_tensor2_unnormalized')
                # Generate gamma
                from TProd3 import tempered_softmax
                l_tensor2 = tempered_softmax(
                    [l_tensor2_unnormalized, gamma], name="l_tensor2")
                print("Very suspicious lack of a nonlinearity, I refuse to run. See ya!")
                sys.exit()

            else:
                l_tensor2 = TensorProdFactLayer((l_shp2, l_input_regions_cell), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
                    vocab), nonlinearity=softmax, name='l_tensor2')

            l_region_feedback = lasagne.layers.ExpressionLayer(
                l_tensor2, lambda X: X.sum(3), output_shape='auto', name='l_region')  # sum over
            l_region_feedback_shape = l_region_feedback

        if CFG['REVIEW_ATTENTION']:
            # TODO Thomas: Test this option, try a more powerfull one with a
            # rnn instead of a feed forward net

            l_region_feedback_shape_reshaped = ReshapeLayer(
                l_region_feedback_shape, ([0], [2]), name='l_region_feedback_shape_reshaped')
            l_input_review_regions = ConcatLayer(
                [l_region_feedback_shape_reshaped, l_cell_input], axis=1, name='l_input_review_regions')
            # TODO HARD CODED CONSTANT BEWARE, test sigmoid instead of ReLU
            l_reviewed_regions_intermediate = DenseLayer(
                l_input_review_regions, num_units=49, nonlinearity=lasagne.nonlinearities.rectify, name='l_reviewed_regions_intermediate')
            l_reviewed_regions = DenseLayer(
                l_reviewed_regions_intermediate, num_units=49, nonlinearity=lasagne.nonlinearities.softmax, name='l_reviewed_regions')

        l_weighted_region = WeightedSumLayer(
            [l_input_regions_cell, l_region_feedback_shape], name='l_weighted_region')
        if CFG['FEEDBACK'] == 0:  # none
            l_cell_concat = l_cell_input
        elif CFG['FEEDBACK'] == 1:
            l_cell_concat = ConcatLayer(
                [l_cell_input, l_region_feedback], axis=1, name='l_cell_concat')
        elif CFG['FEEDBACK'] == 2:
            l_cell_concat = ConcatLayer(
                [l_cell_input, l_weighted_region], axis=1, name='l_cell_concat')
        elif CFG['FEEDBACK'] == 3:
            l_cell_concat = ConcatLayer(
                [l_cell_input, l_weighted_region, l_region_feedback], axis=1, name='l_cell_concat')

        elif CFG['FEEDBACK'] == 4:

            # In this version of the feedback we input the conv5 representation  of the image, with each region weighted by it's corresponding
            # probability. This "new" image then goes through two conv/maxpooling layers to get a reduced dimensionality with a
            # method that takes advantage of the 2D structure of the image.

            # weights each region of the image by the associated weight and restores the 2D structure of the image
            if CFG['IMGFEEDBACK_MECHANISM'] == 'highres':
                # l_region_feedback_shape = lasagne.layers.
                from TProd_devel import WeightedImageLayer
                l_weighted_image = WeightedImageLayer(
                    [l_input_regions_cell_highres, l_region_feedback_shape], upsample=True, name='l_weighted_image')
                l_weighted_image_reshaped = ReshapeLayer(
                    l_weighted_image, ([0], [1], 14, 14), name='l_weighted_image_reshaped')
            else:
                from TProd3 import WeightedImageLayer
                # weights each region of the image by the associated weight.
                l_weighted_image = WeightedImageLayer(
                    [l_input_regions_cell, l_region_feedback_shape], name='l_weighted_image')
                # restores the 2D structure of the image
                l_weighted_image_reshaped = ReshapeLayer(
                    l_weighted_image, ([0], [1], 7, 7), name='l_weighted_image_reshaped')

            # Series of convolution/max pooling with the same number of feature # maps.  NOTE that the names of the layer do not contain "pool" or
            # "conv" but rather "po" or "co" for retrocompatibility with our # old method of selecting RNN layers.
            if CFG['IMGFEEDBACK_MECHANISM'] == 'highres':
                # TODO This might be suboptimal.
                l_weighted_image_conv_reduced = lasagne.layers.MaxPool2DLayer(
                    l_weighted_image_reshaped, (2, 2), name='l_weighted_image_conv_reduced')
                l_feedback_co1 = lasagne.layers.Conv2DLayer(
                    incoming=l_weighted_image_conv_reduced, num_filters=512, filter_size=(3, 3), pad='same', name='l_feedback_co1')
            else:
                l_feedback_co1 = lasagne.layers.Conv2DLayer(
                    incoming=l_weighted_image_reshaped, num_filters=512, filter_size=(3, 3), pad='same', name='l_feedback_co1')

            if CFG['DROP_IMFEEDBACK'] == 1:
                l_feedback_po1 = lasagne.layers.MaxPool2DLayer(
                    l_feedback_co1, (2, 2), name='l_feedback_po1')
                l_feedback_po1_droped = lasagne.layers.DropoutLayer(
                    l_feedback_po1, p=0.5,)
                l_feedback_co2 = lasagne.layers.Conv2DLayer(
                    incoming=l_feedback_po1_droped, num_filters=512, filter_size=(3, 3), pad='same', name='l_feedback_co2')
                l_feedback_po2 = lasagne.layers.MaxPool2DLayer(
                    l_feedback_co2, (2, 2), name='l_feedback_po2')
                # Remove trailing dimensions and concatenate with state vector
                l_feedback_po2_reshaped = ReshapeLayer(
                    l_feedback_po2, ([0], [1]), name='l_feedback_po2_reshaped')
                l_cell_concat = ConcatLayer(
                    [l_cell_input, l_feedback_po2_reshaped], axis=1, name='l_cell_concat')

            else:
                l_feedback_po1 = lasagne.layers.MaxPool2DLayer(
                    l_feedback_co1, (2, 2), name='l_feedback_po1')
                l_feedback_co2 = lasagne.layers.Conv2DLayer(
                    incoming=l_feedback_po1, num_filters=512, filter_size=(3, 3), pad='same', name='l_feedback_co2')
                l_feedback_po2 = lasagne.layers.MaxPool2DLayer(
                    l_feedback_co2, (2, 2), name='l_feedback_po2')
                # Remove trailing dimensions
                l_feedback_po2_reshaped = ReshapeLayer(
                    l_feedback_po2, ([0], [1]), name='l_feedback_po2_reshaped')
                # Concatenate with the state vector.
                l_cell_concat = ConcatLayer(
                    [l_cell_input, l_feedback_po2_reshaped], axis=1, name='l_cell_concat')

        l_gru = GRUMemoryLayer(CFG['EMBEDDING_SIZE'],
                               l_cell_concat, l_prev_gru, name='l_gru')
        # if CFG['USE_RNN_DROPOUT']:
        l_dropout_output = lasagne.layers.DropoutLayer(
            l_gru, p=CFG['RNN_DROPOUT'], name='l_dropout_output')
        l_shp1 = ReshapeLayer(
            l_dropout_output, ([0], 1, [1]), name='l_shp1')

        if CFG['DENSITY_TEMPERING']:
            if CFG['DISSECT'] != 'No':
                print(
                    "Dissect should not be used with density tempering. Now exiting, bye.")
                sys.exit()
            l_gamma = DenseLayer(
                l_shp1, num_units=1, name='l_gamma')
            l_gamma_shp = ReshapeLayer(
                l_gamma, ([0], [1], 1, 1))
            from TProd3 import TensorTemperatureLayer
            l_tensor = TensorTemperatureLayer((l_shp1, l_input_regions_cell, l_gamma_shp), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
                vocab), nonlinearity=lasagne.nonlinearities.softmax, name='l_tensor')
        else:
            if CFG['DENSITY_TEMPERING']:
                print(
                    "Dissect should not be used with density tempering. Now exiting, bye.")
                sys.exit()
            if CFG['DISSECT'] == 'wr':
                l_tensor = TensorProdFactLayer((l_shp1, l_input_regions_cell), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'],
                                               dim_w=len(vocab), nonlinearity=softmax, name='l_tensor', W_hr='skip', b_hr='skip')
            elif CFG['DISSECT'] == 'rs':
                l_tensor = TensorProdFactLayer((l_shp1, l_input_regions_cell), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'],
                                               dim_w=len(vocab), nonlinearity=softmax, name='l_tensor', W_rw='skip', b_rw='skip')
            else:
                l_tensor = TensorProdFactLayer((l_shp1, l_input_regions_cell), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
                    vocab), nonlinearity=softmax, name='l_tensor')

        # Ablation of the tensor to show usefullness of some terms.

        l_word = lasagne.layers.ExpressionLayer(l_tensor, lambda X: X.sum(
            2), output_shape='auto', name='l_word')  # sum over
        if CFG['TENSOR_COND_WORD']:
            from TProd3 import RegionCondWordLayer
            l_region = RegionCondWordLayer(l_tensor, name='l_region')
        else:
            l_region = lasagne.layers.ExpressionLayer(l_tensor, lambda X: X.sum(
                3), output_shape='auto', name='l_region')  # sum over regions
        l_region = ReshapeLayer(
            l_region, ([0], [2]), name='l_region')
        from collections import OrderedDict
        if CFG['TENSOR_TIED']:
            memory_dict = OrderedDict(
                [(l_gru, l_prev_gru), (l_region, l_region_feedback)])
        else:
            memory_dict = OrderedDict([(l_gru, l_prev_gru)])
        from agentnet.agent import Recurrence
        if CFG['IMGFEEDBACK_MECHANISM'] == 'highres':
            l_rec = Recurrence(
                input_nonsequences={l_input_regions_cell: l_input_regions,
                                    l_input_regions_cell_highres: l_input_regions_highres},
                # we use out previously defined dictionary to update recurrent network parameters
                state_variables=memory_dict,
                # we feed in reference sequence into "prev letter" input layer, tick by tick along the axis=1.
                input_sequences={l_cell_input: l_dropout_input},
                # we track agent would-be actions and probabilities
                tracked_outputs=l_word,
                n_steps=CFG['SEQUENCE_LENGTH'],
                # finally, we define an optional batch size param
                #(if omitted, it will be inferred from inputs or initial value providers if there are any)
                batch_size=CFG['BATCH_SIZE'] * 5,
            )
        else:
            l_rec = Recurrence(
                input_nonsequences={l_input_regions_cell: l_input_regions},
                # we use out previously defined dictionary to update recurrent network parameters
                state_variables=memory_dict,
                # we feed in reference sequence into "prev letter" input layer, tick by tick along the axis=1.
                input_sequences={l_cell_input: l_dropout_input},
                # we track agent would-be actions and probabilities
                tracked_outputs=l_word,
                n_steps=CFG['SEQUENCE_LENGTH'],
                # finally, we define an optional batch size param
                #(if omitted, it will be inferred from inputs or initial value providers if there are any)
                batch_size=CFG['BATCH_SIZE'] * 5,
            )

        l_gru_states, l_out = l_rec.get_sequence_layers()
        #}}}

    d = {}

    if CFG['START_FROM'] != '':
        d = pickle.load(open(CFG['START_FROM']))
        d2 = d.copy()
        #from save_layers import set_param_dict
        if CFG['CONVERT_TRANSFORMER']:
            d['param values']['l_sel_region.W'] = d['param values']['l_add_con.W'].reshape(
                (CFG['REGION_SIZE'], CFG['REGION_SIZE'] * 3 * 3)).swapaxes(0, 1)
            d['param values']['l_sel_region.b'] = d['param values']['l_add_con.b']
        relaxed = False
        if CFG['TRANS_USE_PRETRAINED']:
            relaxed = True
        set_param_dict(l_out, d['param values'], relax=relaxed)

    init_epoc = 0  # iteration to start with
    if CFG['RESTART']:
        print "All Layers", [a.name for a in lasagne.layers.get_all_layers(l_out)]
        file_name = '%s/%s_*.pkl' % (CFG['FILE_NAME'], CFG['MODE'])
        from glob import glob
        list_files = glob(file_name)
        last_epoc = 0
        if list_files != []:
            last_epoc = np.array([int(x.split('_')[-1].split('.')[0])
                                  for x in list_files]).argmax()
            d = pickle.load(open(list_files[last_epoc]))
            print "Restart Loaded", list_files[last_epoc]
            CFG, loss_tr, loss_val = load_config(CFG, d['config'])
            init_epoc = int(list_files[last_epoc].split('_')[-1].split('.')[0])
            print "Starting from epoc", init_epoc + 1
            from save_layers import set_param_dict
            set_param_dict(l_out, d['param values'])
        else:
            print 'No previous snapshot found, starting from scratch!'

        if CFG['SAVE_PARTIAL'] != -1:
            load_name = '%s/%s_%d.pkl.partial' % (
                CFG['FILE_NAME'], CFG['MODE'], init_epoc + 1)
            if os.path.isfile(load_name):
                print "Loading partial saving", load_name
                d = pickle.load(open(load_name))
                if d['config']['SAVE_PARTIAL'] != CFG['SAVE_PARTIAL']:
                    print('Error the current save_partial is different than the saved!')
                    sys.exit()
                print('Starting from iteration:',
                      d['partial'] * CFG['SAVE_PARTIAL'])
                print('Computed as partial:',
                      d['partial'], '*', 'save_partial:', CFG['SAVE_PARTIAL'])
                CFG, loss_tr, loss_val = load_config(CFG, d['config'])
                set_param_dict(l_out, d['param values'])
            else:
                print('No partial save found!')

    if params['print']:
        print CFG
        import sys
        fsfd
        sys.exit()

    # cnn feature vector
    x_cnn_sym = T.matrix()
    # cnn regon feature vector
    x_conv_sym = T.tensor3()
    x_img_sym = T.tensor4()
    x_img2_sym = T.tensor4()
    x_boxes_sym = T.matrix()

    # sentence encoded as sequence of integer word tokens
    x_sentence_sym = T.imatrix()

    # mask defines which elements of the sequence should be predicted
    mask_sym = T.imatrix()

    # ground truth for the RNN output
    y_sentence_sym = T.imatrix()

    #x_hid_inp = T.matrix()
    x_feedback = T.matrix()

    lr = theano.shared(np.array(CFG['LR'], dtype=np.float32))

    if CFG['MODE'] == 'normal' or CFG['MODE'] == 'lrcn' or CFG['MODE'] == 'tensor':

        if CFG['PROPOSALS'] == 5:
            output = lasagne.layers.get_output(l_out, {
                l_input_sentence: x_sentence_sym,
                l_input_img: x_img_sym,
                l_input_img2: x_img2_sym,
            }, deterministic=False)
        else:
            output = lasagne.layers.get_output(l_out, {
                l_input_sentence: x_sentence_sym,
                l_input_img: x_img_sym,
            }, deterministic=False)

    if CFG['MODE'] == 'transformer':
        if CFG['TRANS_FEEDBACK']:
            output = lasagne.layers.get_output(l_out, {
                l_input_sentence: x_sentence_sym,
                l_input_img: x_img_sym,
            }, deterministic=False)
        else:
            output = lasagne.layers.get_output(l_out, {
                l_input_sentence: x_sentence_sym,
                l_input_img: x_img_sym,
            }, deterministic=False)

    elif CFG['MODE'] == 'tensor-feedback':
        output = lasagne.layers.get_output(l_out, {
            l_input_sentence: x_sentence_sym,
            l_input_cnn: x_cnn_sym,
            l_input_regions: x_conv_sym,
        })

    elif CFG['MODE'] == 'tensor-feedback2':
        if CFG['PROPOSALS'] in [3, 4]:
            output = lasagne.layers.get_output(l_out, {
                l_input_sentence: x_sentence_sym,
                l_input_img: x_img_sym,
                l_input_img2: x_img2_sym,
                l_boxes: x_boxes_sym,
            }, deterministic=False)
        else:
            output = lasagne.layers.get_output(l_out, {
                l_input_sentence: x_sentence_sym,
                l_input_img: x_img_sym,
            })

    loss = T.mean(calc_cross_ent(output, mask_sym, y_sentence_sym))

    if CFG['REG_H']:
        lstm_out = lasagne.layers.get_output(l_lstm, {
            l_input_sentence: x_sentence_sym,
            l_input_cnn: x_cnn_sym,
        })

        reg = T.mean((T.sqrt(
            T.sum(lstm_out[:, 0:-1, :]**2, 2)) - T.sqrt(T.sum(lstm_out[:, 1:, :]**2, 2)))**2)
        loss = loss + reg * CFG['REG_H']

    MAX_GRAD_NORM = CFG['GRAD_CLIP']
    if CFG['TRANS_REGLEARN'] > 0:
        all_params_reg = lasagne.layers.get_all_params(
            l_out, trainable=True, reglearn=True)
        # l2 reg
        #loss = loss + CFG['TRANS_REGLEARN']*T.sum([T.sum(p**2) for p in all_params_reg])
        # l1 reg
        loss = loss + CFG['TRANS_REGLEARN'] * \
            T.sum([T.sum(T.abs_(p)) for p in all_params_reg])

    # new way of selecting parts of the network
    if CFG['TRAIN_ONLY_CNN']:
        all_params = lasagne.layers.get_all_params(l_input_cnn, trainable=True)
    elif CFG['TRAIN_ONLY_RNN'] or CFG['CNN_SLOWLEARN']:
        all_params = lasagne.layers.get_all_params(
            l_out, trainable=True, conv_net=False, conv_net2=False)
    else:
        all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    all_grads = T.grad(loss, all_params)
    all_grads = [T.clip(g, -CFG['GRAD_CLIP'], CFG['GRAD_CLIP'])
                 for g in all_grads]  # 1,-1
    norms = T.sqrt([T.sum(tensor**2) for tensor in all_grads])
    all_grads, norm = lasagne.updates.total_norm_constraint(
        all_grads, MAX_GRAD_NORM, return_norm=True)

    updates = lasagne.updates.adam(
        all_grads, all_params, learning_rate=lr)  # 0.001)# 0.00001

    if CFG['CNN_SLOWLEARN']:
        trans_params = lasagne.layers.get_all_params(
            l_out, conv_net=True, conv_net2=True)
        trans_grads = T.grad(loss, trans_params)
        trans_grads = [T.clip(g, -CFG['GRAD_CLIP'], CFG['GRAD_CLIP'])
                       for g in trans_grads]  # 1,-1
        trans_grads, norm = lasagne.updates.total_norm_constraint(
            trans_grads, MAX_GRAD_NORM, return_norm=True)

        trans_updates = lasagne.updates.adam(
            trans_grads, trans_params, learning_rate=lr * 0.1)  # 0.001)# 0.00001
        updates.update(trans_updates)

    if CFG['TRANS_SLOWLEARN']:
        trans_params = lasagne.layers.get_all_params(l_out, slowlearn=True)
        trans_grads = T.grad(loss, trans_params)
        trans_grads = [T.clip(g, -CFG['GRAD_CLIP'], CFG['GRAD_CLIP'])
                       for g in trans_grads]
        trans_grads, norm = lasagne.updates.total_norm_constraint(
            trans_grads, MAX_GRAD_NORM, return_norm=True)

        trans_updates = lasagne.updates.adam(
            trans_grads, trans_params, learning_rate=lr * 0.1)
        updates.update(trans_updates)

    if (CFG['RESTART'] or CFG['START_FROM'] != '') and d.has_key('adam values'):
        if len(updates) == len(d['adam values']):
            print('Loading the adam parameters for a better restart!')
            for u, v in zip(updates, d['adam values']):
                u.set_value(v)
        elif len(updates) > len(d['adam values']):
            print('Loading the adam parameters of the RNN!')
            # assume that you start from a previous training
            if CFG['TRAIN_ONLY_RNN'] == False and CFG['START_FROM'] != '':
                if d2['config']['TRANS_DENSE_NOLEARN']:
                    all_params2 = lasagne.layers.get_all_params(
                        l_out, trainable=True, conv_net=False, conv_net2=False, nolearn=False)
                else:
                    all_params2 = lasagne.layers.get_all_params(
                        l_out, trainable=True, conv_net=False, conv_net2=False)
            else:
                all_params2 = lasagne.layers.get_all_params(
                    l_out, trainable=True, conv_net=False, conv_net2=False, locnet=False, nolearn=False)
            all_grads2 = T.grad(loss, all_params2)
            all_grads2 = [T.clip(g, -CFG['GRAD_CLIP'], CFG['GRAD_CLIP'])
                          for g in all_grads2]  # 1,-1
            all_grads2, norm = lasagne.updates.total_norm_constraint(
                all_grads2, MAX_GRAD_NORM, return_norm=True)
            updates2 = lasagne.updates.adam(
                all_grads2, all_params2, learning_rate=lr)  # 0.001)# 0.00001
            assert(len(updates2) == len(d['adam values']))
            for u, v in zip(updates2, d['adam values']):
                u.set_value(v)
        else:
            print('Changed some configuration, I cannot load the old adam parameters!')

    t = time.time()
    print("Compiling theano functions...")

    f_train = theano.function([x_img_sym, x_img2_sym, x_sentence_sym, mask_sym, y_sentence_sym, x_boxes_sym],
                              [loss, norm, norms],
                              updates=updates, on_unused_input='warn')  # ,mode=theano.compile.mode.Mode(optimizer=None)                             )

    print("f_train has been compiled")

    f_val = theano.function([x_img_sym, x_img2_sym, x_sentence_sym, mask_sym,
                             y_sentence_sym, x_boxes_sym], loss, on_unused_input='warn')

    print("Functions compiled in %d secs" % (time.time() - t))

    import time
    norm = 0
    norms = 0
    decay_factor = 0.2
    loss_train = 0
    if CFG['PROPOSALS'] == 1:
        im_size = 224
    elif CFG['PROPOSALS'] > 1:
        im_size = CFG['IM_SIZE']
    else:
        im_size = -1  # tell the data generator to not prepare images for proposals
    samples = 0
    decay_epocs = 1  # CFG['LR_DECAY_EPOCS']
    if params['small_dataset']:
        seval = 100
    else:
        seval = 1000
    # recover correct lr
    if init_epoc > 0:
        for epoc in range(0, init_epoc):
            if epoc >= CFG['LR_DECAY_START']:
                decay_epocs -= 1
                if decay_epocs == 0:
                    lr = lr * decay_factor
                    decay_epocs = CFG['LR_DECAY_EPOCS']
        print('Learning rate set to {}'.format(lr.eval()))

    if CFG['SAVE_PARTIAL'] != -1:
        if d.has_key('partial'):
            start_from = d['partial'] * CFG['SAVE_PARTIAL'] * \
                seval / CFG['BATCH_SIZE'] * CFG['BATCH_SIZE']
            samples = start_from
        else:
            start_from = 0
    else:
        start_from = 0

    for epoc in range(init_epoc, CFG['EPOCS']):
        t = time.time()
        partial_count = 0
        # t1=time.time()

        if CFG['SAVE_PARTIAL'] != -1:
            save_name = ''
            if CFG.has_key('SET_SEED'):
                random.seed(init_epoc)
            else:
                random.seed(init_epoc + params['set_seed'])
            if d.has_key('partial'):
                partial_count = d['partial']

        if 0:
            parameters = CFG, dbtrain, CFG['BATCH_SIZE'], im_size, word_to_index, True, start_from, CFG['USE_FLIP']
            import pdb
            pdb.set_trace()

        for img_id, x_img, x_img2, x_sent, y_sent, mask, x_boxes in batch_gen(CFG, dbtrain, CFG['BATCH_SIZE'], im_size, word_to_index, shuffle=True, start_from=start_from, use_flip=CFG['USE_FLIP']):

            if samples % (seval / CFG['BATCH_SIZE'] * CFG['BATCH_SIZE']) == 0:
                print('Epoc {}, Samples {}, loss_train: {}, norm: {} lr: {} time: {}'.format(
                    epoc + 1, samples, loss_train, norm, lr.eval(), time.time() - t))
                if params['verbose']:
                    if CFG['TRAIN_ONLY_CNN']:
                        print('Layers {}, norms: {}'.format(
                            lasagne.layers.get_all_params(l_input_cnn, trainable=True), norms))
                    else:
                        print('Layers {}, norms: {}'.format(
                            lasagne.layers.get_all_params(l_out, trainable=True), norms))
                    if 1:
                        try:
                            param_values = get_param_dict_tied(
                                l_out, new_style=True)
                            print "W_hw", np.sqrt(np.sum(param_values['l_tensor.W_hw']**2))
                            print "W_hr", np.sqrt(np.sum(param_values['l_tensor.W_hr']**2))
                            print "W_rw", np.sqrt(np.sum(param_values['l_tensor.W_rw']**2))
                        except:
                            pass
                # try:
                batch_loss_val = []
                batch_loss_tr = []
                count = 0
                for vimg_id, vx_img, vx_img2, vx_sent, vy_sent, vmask, vx_boxes in batch_gen(CFG, dbtrain, CFG['BATCH_SIZE'], im_size, word_to_index, shuffle=False):
                    batch_loss_tr.append(
                        f_val(vx_img, vx_img2, vx_sent, vmask, vy_sent, vx_boxes))

                    # Check the sentences obtained + go see if the img_id correspond in the image folder.
                    if 0:
                        print(
                            "\n Training sentences: ----------------------------------")
                        for sent_index in range(30):
                            print(" ".join([index_to_word[vx_sent[sent_index, i]]
                                            for i in range(21)][1:]).split('#END')[0])
                        import pdb
                        pdb.set_trace()
                        print("\n\n")

                    # Checking out what sentences f_train and f_val produce: make them return 'output' and find how to get the mode of the distribution.
                    count += 1
                    if count * CFG['BATCH_SIZE'] >= 100:
                        break
                    # train
                t = time.time()
                count = 0
                if not (params['dataset'] == 'flickr') and params['small_dataset']:
                    for vimg_id, vx_img, vx_img2, vx_sent, vy_sent, vmask, vx_boxes in batch_gen(CFG, dbval, CFG['BATCH_SIZE'], im_size, word_to_index, shuffle=False):
                        batch_loss_val.append(
                            f_val(vx_img, vx_img2, vx_sent, vmask, vy_sent, vx_boxes))
                        count += 1
                        if count * CFG['BATCH_SIZE'] >= 100:
                            break
                loss_val.append(np.mean(batch_loss_val))
                loss_tr.append(np.mean(batch_loss_tr))
                print('Tr loss: {}, time: {}'.format(
                    loss_tr[-1], time.time() - t))
                print('Val loss: {}, time: {}'.format(
                    loss_val[-1], time.time() - t))
                if params['plot_loss']:
                    import pylab
                    pylab.figure(1)
                    pylab.clf()
                    pylab.plot(loss_tr, 'b', label='Train')
                    pylab.plot(loss_val, 'g', label='Validation')
                    pylab.xlabel('Iterations')
                    pylab.ylabel('Loss')
                    pylab.legend()
                    pylab.grid()
                    pylab.draw()
                    # pylab.show()
                    if not os.path.exists(CFG['FILE_NAME']):
                        os.makedirs(CFG['FILE_NAME'])
                    # print('%s/%s.pdf'%(CFG['FILE_NAME'],CFG['FILE_NAME'].split('/')[-1]))
                    pylab.savefig(
                        '%s/%s.pdf' % (CFG['FILE_NAME'], CFG['FILE_NAME'].split('/')[-1]))
                    # raw_input()
                t = time.time()

            loss_train, norm, norms = f_train(
                x_img, x_img2, x_sent, mask, y_sent, x_boxes)
            if np.isnan(loss_train):
                raise ValueError('Loss is Nan!!!')
            samples += CFG['BATCH_SIZE']

            # This was done as an original fix for finetuning, I think it is
            # useless now, but harmless and I keep it to be compatible with the
            # experiments that are still running.
            param_values = get_param_dict_tied(l_out, new_style=True)
            if CFG['SAVE_PARTIAL'] != -1 and samples % (CFG['SAVE_PARTIAL'] * seval / CFG['BATCH_SIZE'] * CFG['BATCH_SIZE']) == 0:
                print('Saving partial epoc')
                save_name = save_epoc(epoc, updates, param_values, vocab, word_to_index,
                                      index_to_word, CFG, loss_tr, loss_val, partial=partial_count + 1)
                partial_count += 1

        start_from = 0

        #param_values = lasagne.layers.get_all_param_values(l_out)
        if epoc >= CFG['LR_DECAY_START']:
            decay_epocs -= 1
            if decay_epocs == 0:
                lr = lr * decay_factor
                decay_epocs = CFG['LR_DECAY_EPOCS']

        param_values = get_param_dict_tied(l_out, new_style=True)

        if (CFG['DATASET'] == 'coco') or ((CFG['DATASET'] == 'flickr') and epoc % params['epoc_interval'] == 0):
            save_epoc(epoc, updates, param_values, vocab,
                      word_to_index, index_to_word, CFG, loss_tr, loss_val)
        if CFG['SAVE_PARTIAL'] != -1:
            if os.path.isfile(save_name):
                print('Removed Partial Saving')
                os.remove(save_name)
    print("Total Training Time :{}h".format(
        (time.time() - starting_time) / 3600.))
