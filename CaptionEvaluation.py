# coding: utf-8
# vim:fdm=marker

# # Image Captioning with LSTM

try:
    import os
    import subprocess
    gpu_id = subprocess.check_output('gpu_getIDs.sh', shell=True)
    os.environ["THEANO_FLAGS"] = 'device=gpu%s' % gpu_id
    print(os.environ["THEANO_FLAGS"])
except:
    pass

import argparse
import numpy as np
import theano
from lasagne.utils import floatX
import lasagne
import sys
import os
import scipy
#import skimage.transform


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


import theano.tensor as T
# use this lines for fast compilation and theano debug
# theano.config.optimizer='None'#'fast_compile'
# theano.config.exception_verbosity='high'

import matplotlib
if not run_from_ipython():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
import pickle
from bcolors import bcolors

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import dropout
from lasagne.layers import ReshapeLayer
from lasagne.layers import DimshuffleLayer
from lasagne.layers import TransformerLayer
from lasagne.layers import ExpressionLayer
from lasagne.nonlinearities import softmax
from Transformer import TranslateLayer
from Transformer import MultiTransformerLayer
from agentnet.memory import GRUMemoryLayer
from TProd3 import TensorProdFactLayer

from RNNTraining import build_loc_net
from flickr_prepareData import load_flickr
from build_eval import buildNetwork


def setParamNetwork(CFG, l_out, l_cnn_embedding, l_sentence_embedding, l_out_reg, d, show_layers=True, relax_check=False):
    # {{{
    from save_layers import set_param_dict, check_names, check_init
    set_par = 0
    if CFG.has_key('NEW_SAVE') and CFG['NEW_SAVE']:
        check_names(l_out, new_style=True)
        check_names(l_cnn_embedding, new_style=True)
        check_names(l_sentence_embedding, new_style=True)
        check_names(l_out_reg, new_style=True)
        if 0:
            d['param values']['l_loc.W'][...] = 0
            d['param values']['l_loc.b'][...] = 0
            if CFG['TRANS_ADD_BIG_PROPOSALS']:
                d['param values']['l_loc_big.W'][...] = 0
                d['param values']['l_loc_big.b'][...] = 0
        relax = False
        if CFG.has_key('FORCE_TRANSFORMER') and CFG['FORCE_TRANSFORMER']:
            relax = True
        used1 = set_param_dict(
            l_out, d['param values'], show_layers=show_layers, relax=relax)
        used2 = set_param_dict(
            l_cnn_embedding, d['param values'], show_layers=show_layers)
        used3 = set_param_dict(l_sentence_embedding,
                               d['param values'], show_layers=show_layers)
        used4 = set_param_dict(
            l_out_reg, d['param values'], show_layers=show_layers)
        used = set()
        used.update(used1, used2, used3, used4)
        if set(d['param values'].keys()) != used:
            print('Warning, the number of loaded parameters is not fully used!')
            print(set(d['param values'].keys()).difference(used))
            if not relax_check:
                raw_input()

        if CFG.has_key('RELAX_CHECK_INIT'):
            relax_check = CFG['RELAX_CHECK_INIT']
            assert(relax_check == 0 or relax_check ==
                   1), "relax_check is missused"
        else:
            relax_check = False

        if not CFG.has_key('FORCE_TRANSFORMER') or not CFG['FORCE_TRANSFORMER']:
            check_init(l_out, relax_check)

        check_init(l_cnn_embedding, relax_check)
        check_init(l_sentence_embedding, relax_check)
        if not CFG.has_key('FORCE_PROPOSALS') or not CFG['FORCE_PROPOSALS']:
            check_init(l_out_reg, relax_check)
    else:
        check_names(l_out, new_style=True)
        # remove initial part, just a patch, I need to fix save_param
        words = ['YetAnotherRecurrence_', 'l_tensor',
                 'l_cnn_embedding_', 'l_sentence_embedding_']
        for ss in words:
            for l in d['param values'].keys():
                pos = l.find(ss)
                if pos != -1:
                    d['param values'][l[pos + len(ss):]] = d['param values'][l]

        set_param_dict(l_out, d['param values'])
        set_param_dict(l_cnn_embedding, d['param values'])
        set_param_dict(l_sentence_embedding, d['param values'])
    if d['param values'].has_key('l_tensor.W_hw'):
        print 'W_hw', np.sqrt(np.sum(d['param values']['l_tensor.W_hw']**2))
    else:
        print 'W_hw', np.sqrt(np.sum(d['param values']['l_decoder.W']**2))
    if d['param values'].has_key('l_tensor.W_rw'):
        print 'W_rw', np.sqrt(np.sum(d['param values']['l_tensor.W_rw']**2))
    if d['param values'].has_key('l_tensor.W_hr'):
        print 'W_hr', np.sqrt(np.sum(d['param values']['l_tensor.W_hr']**2))
    # }}}


def compileNetwork(CFG, net):
    # {{{
    l_loc1 = net['loc1']
    l_input_loc = net['input_loc']
    l_loc = net['loc']
    l_prev_gru = net['prev']
    l_cell_input = net['input']
    l_gru = net['gru']
    l_input_sentence = net['sent']
    l_input_img = net['img']
    l_input_img2 = net['img2']
    l_out = net['out']
    l_cnn_embedding = net['cnn']
    l_sentence_embedding = net['sent_emb']
    l_input_reg = net['reg']
    l_out_reg = net['out_reg']
    l_decoder = net['decoder']
    l_region = net['reg_feedback']
    l_region_feedback = net['reg_feedback2']
    l_boxes = net['boxes']
    l_conv = net['conv']
    l_sel_region2 = net['sel_region2']
    l_weighted_region_prev = net['weighted_regions_prev']
    l_weighted_region = net['weighted_regions']

    mask_sym = T.imatrix()
    x_cnn_sym = T.matrix()
    x_sentence_sym = T.imatrix()
    x_rnn = T.matrix()
    x_cell = T.matrix()
    x_conv = T.tensor4()
    x_img = T.tensor4()
    x_img2 = T.tensor4()
    x_hid = T.matrix()
    x_region = T.matrix()
    x_boxes = T.matrix()

    if CFG['CNN_FINE_TUNE']:
        if CFG['MODE'] == 'normal' or CFG['MODE'] == 'lrcn':
            emb_cnn = lasagne.layers.get_output(l_cnn_embedding, {
                l_input_img: x_img,
            }, deterministic=True)
            f_cnn = theano.function([x_img], emb_cnn, on_unused_input='warn')
        else:  # tensor
            if CFG['PROPOSALS'] == 1:
                emb_cnn = lasagne.layers.get_output([l_cnn_embedding, l_out_reg], {
                    l_input_img: x_img,
                    l_boxes: x_boxes,
                }, deterministic=True)
                f_cnn = theano.function(
                    [x_img, x_boxes], emb_cnn, on_unused_input='warn')
            elif CFG['PROPOSALS'] > 1:
                emb_cnn = lasagne.layers.get_output([l_cnn_embedding, l_out_reg, l_conv], {
                    l_input_img: x_img,
                    l_input_img2: x_img2,
                    l_boxes: x_boxes,
                }, deterministic=True)
                f_cnn = theano.function(
                    [x_img, x_img2, x_boxes], emb_cnn, on_unused_input='warn')
            else:
                emb_cnn = lasagne.layers.get_output([l_cnn_embedding, l_out_reg], {
                    l_input_img: x_img,
                }, deterministic=True)
                f_cnn = theano.function(
                    [x_img], emb_cnn, on_unused_input='warn')

    else:
        emb_cnn = lasagne.layers.get_output(l_cnn_embedding, {
            l_input_cnn: x_cnn_sym,
        }, deterministic=True)
        f_cnn = theano.function([x_cnn_sym], emb_cnn, on_unused_input='warn')

    emb_sent = lasagne.layers.get_output(l_sentence_embedding, {
        l_input_sentence: x_sentence_sym,
    }, deterministic=True)

    if CFG['MODE'] == 'normal':
        output = lasagne.layers.get_output([l_out, l_gru, l_gru], {
            l_cell_input: x_rnn,
            l_prev_gru: x_hid,
        }, deterministic=True)
    if CFG['MODE'] == 'lrcn':
        output = lasagne.layers.get_output([l_out, l_gru, l_gru], {
            l_cell_input: x_rnn,
            l_input_reg: x_conv,
            l_prev_gru: x_hid,
        }, deterministic=True)
    elif CFG['MODE'] == 'tensor':
        if CFG['VISUALIZE'] > 0:
            if CFG['PROPOSALS'] > 1:
                output = lasagne.layers.get_output([l_out, l_gru, l_decoder], {
                    l_cell_input: x_rnn,
                    l_input_reg: x_conv,
                    l_prev_gru: x_hid,
                    l_boxes: x_boxes,
                    l_input_img2: x_img2,
                }, deterministic=True)
            else:
                output = lasagne.layers.get_output([l_out, l_gru, l_decoder], {
                    l_cell_input: x_rnn,
                    l_input_reg: x_conv,
                    l_prev_gru: x_hid,
                }, deterministic=True)
        else:
            if CFG['PROPOSALS'] > 1:
                output = lasagne.layers.get_output([l_out, l_gru, l_gru], {
                    l_cell_input: x_rnn,
                    l_input_reg: x_conv,
                    l_prev_gru: x_hid,
                    l_boxes: x_boxes,
                    l_input_img2: x_img2,
                }, deterministic=True)
            else:
                output = lasagne.layers.get_output([l_out, l_gru, l_gru], {
                    l_cell_input: x_rnn,
                    l_input_reg: x_conv,
                    l_prev_gru: x_hid,
                }, deterministic=True)
    elif CFG['MODE'] == 'transformer':
        if CFG['VISUALIZE'] > 0:
            output = lasagne.layers.get_output([l_out, l_gru, l_weighted_region, l_loc, l_decoder], {  # ,l_input_loc,l_loc1],{#,l_sel_region2], {#l_loc], {
                l_cell_input: x_rnn,
                l_input_reg: x_conv,
                l_prev_gru: x_hid,
                l_weighted_region_prev: x_region,
            }, deterministic=True)
        else:
            if CFG['TRANS_FEEDBACK']:
                output = lasagne.layers.get_output([l_out, l_gru, l_weighted_region], {
                    l_cell_input: x_rnn,
                    l_input_reg: x_conv,
                    l_prev_gru: x_hid,
                    l_weighted_region_prev: x_region,
                }, deterministic=True)
            else:
                output = lasagne.layers.get_output([l_out, l_gru, l_gru], {
                    l_cell_input: x_rnn,
                    l_input_reg: x_conv,
                    l_prev_gru: x_hid,
                }, deterministic=True)

    elif CFG['MODE'] == 'tensor-feedback' or CFG['MODE'] == 'tensor-feedback2':
        if CFG['PROPOSALS'] in [3, 4]:
            output = lasagne.layers.get_output([l_out, l_gru, l_region], {
                l_cell_input: x_rnn,
                l_input_reg: x_conv,
                l_prev_gru: x_hid,
                l_boxes: x_boxes,
                l_input_img2: x_img2,
                l_region_feedback: x_region,
            }, deterministic=True)
        elif CFG['PROPOSALS'] > 1:
            print("Proposals versions other that 3 and 4 are not supported with feedback")
            sys.exit()
        else:
            output = lasagne.layers.get_output([l_out, l_gru, l_region], {
                l_cell_input: x_rnn,
                l_input_reg: x_conv,
                # l_input_cell: x_cell,
                l_prev_gru: x_hid,
                l_region_feedback: x_region,
            }, deterministic=True)

    f = theano.function([x_rnn, x_conv, x_hid, x_region,
                         x_boxes, x_img2], output, on_unused_input='warn')

    f_sent = theano.function(
        [x_sentence_sym], emb_sent, on_unused_input='warn')

    return f_cnn, emb_cnn, f, f_sent
# }}}


def compileEnsemble(CFG, net):
    #{{{1

    l_loc1 = dict()
    l_input_loc = dict()
    l_loc = dict()
    l_prev_gru = dict()
    l_cell_input = dict()
    l_gru = dict()
    l_input_sentence = dict()
    l_input_img = dict()
    l_input_img2 = dict()
    l_out = dict()
    l_cnn_embedding = dict()
    l_sentence_embedding = dict()
    l_input_reg = dict()
    l_out_reg = dict()
    l_decoder = dict()
    l_region = dict()
    l_region_feedback = dict()
    l_boxes = dict()
    l_conv = dict()
    l_sel_region2 = dict()
    l_weighted_region_prev = dict()
    l_weighted_region = dict()

    for i in range(1, len(net) + 1):
        l_loc1[i] = net[i]['loc1']
        l_input_loc[i] = net[i]['input_loc']
        l_loc[i] = net[i]['loc']
        l_prev_gru[i] = net[i]['prev']
        l_cell_input[i] = net[i]['input']
        l_gru[i] = net[i]['gru']
        l_input_sentence[i] = net[i]['sent']
        l_input_img[i] = net[i]['img']
        l_input_img2[i] = net[i]['img2']
        l_out[i] = net[i]['out']
        l_cnn_embedding[i] = net[i]['cnn']
        l_sentence_embedding[i] = net[i]['sent_emb']
        l_input_reg[i] = net[i]['reg']
        l_out_reg[i] = net[i]['out_reg']
        l_decoder[i] = net[i]['decoder']
        l_region[i] = net[i]['reg_feedback']
        l_region_feedback[i] = net[i]['reg_feedback2']
        l_boxes[i] = net[i]['boxes']
        l_conv[i] = net[i]['conv']
        l_sel_region2[i] = net[i]['sel_region2']
        l_weighted_region_prev[i] = net[i]['weighted_regions_prev']
        l_weighted_region[i] = net[i]['weighted_regions']

    mask_sym = T.imatrix()
    x_cnn_sym = T.matrix()
    x_sentence_sym = T.imatrix()
    x_rnn = T.matrix()  # tensor3()
    x_cell = T.matrix()
    x_conv = T.tensor4()
    x_img = T.tensor4()
    x_img2 = T.tensor4()
    x_hid = T.matrix()
    x_region = T.matrix()
    x_boxes = T.matrix()

    emb_cnn = dict()
    emb_sent = dict()
    f_cnn = dict()
    output = dict()
    f = dict()
    f_sent = dict()
    for i in range(1, len(net) + 1):
        if CFG[i]['CNN_FINE_TUNE']:
            if CFG[i]['MODE'] == 'normal' or CFG[i]['MODE'] == 'lrcn':
                emb_cnn[i] = lasagne.layers.get_output(l_cnn_embedding[i], {
                    l_input_img[i]: x_img,
                }, deterministic=True)
                f_cnn[i] = theano.function(
                    [x_img], emb_cnn[i], on_unused_input='warn')
            else:  # tensor
                if CFG[i]['PROPOSALS'] > 1:

                    emb_cnn[i] = lasagne.layers.get_output([l_cnn_embedding[i], l_out_reg[i], l_conv[i]], {
                        l_input_img[i]: x_img,
                        l_input_img2[i]: x_img2,
                        l_boxes[i]: x_boxes,
                    }, deterministic=True)
                    f_cnn[i] = theano.function(
                        [x_img, x_img2, x_boxes], emb_cnn[i], on_unused_input='warn')
                else:
                    emb_cnn[i] = lasagne.layers.get_output([l_cnn_embedding[i], l_out_reg[i]], {
                        l_input_img[i]: x_img,
                    }, deterministic=True)
                    f_cnn[i] = theano.function(
                        [x_img], emb_cnn[i], on_unused_input='warn')

        else:
            emb_cnn[i] = lasagne.layers.get_output(l_cnn_embedding[i], {
                l_input_cnn[i]: x_cnn_sym,
            }, deterministic=True)
            f_cnn[i] = theano.function(
                [x_cnn_sym], emb_cnn[i], on_unused_input='warn')

        emb_sent[i] = lasagne.layers.get_output(l_sentence_embedding[i], {
            l_input_sentence[i]: x_sentence_sym,
        }, deterministic=True)

        if CFG[i]['MODE'] == 'normal':
            output[i] = lasagne.layers.get_output([l_out, l_gru, l_gru], {
                l_cell_input: x_rnn,
                l_prev_gru: x_hid,
            }, deterministic=True)
        if CFG[i]['MODE'] == 'lrcn':
            output[i] = lasagne.layers.get_output([l_out[i], l_gru[i], l_gru[i]], {
                l_cell_input[i]: x_rnn,
                l_input_reg[i]: x_conv,
                l_prev_gru[i]: x_hid,
            }, deterministic=True)

        elif CFG[i]['MODE'] == 'tensor':
            if CFG[i]['VISUALIZE'] > 0:
                if CFG[i]['PROPOSALS'] > 1:
                    output[i] = lasagne.layers.get_output([l_out[i], l_gru[i], l_decoder[i]], {
                        l_cell_input[i]: x_rnn,
                        l_input_reg[i]: x_conv,
                        l_prev_gru[i]: x_hid,
                        l_boxes[i]: x_boxes,
                        l_input_img2[i]: x_img2,
                    }, deterministic=True)
                else:
                    output[i] = lasagne.layers.get_output([l_out[i], l_gru[i], l_decoder[i]], {
                        l_cell_input[i]: x_rnn,
                        l_input_reg[i]: x_conv,
                        l_prev_gru[i]: x_hid,
                    }, deterministic=True)
            else:
                if CFG[i]['PROPOSALS'] > 1:
                    output[i] = lasagne.layers.get_output([l_out[i], l_gru[i], l_gru[i]], {
                        l_cell_input[i]: x_rnn,
                        l_input_reg[i]: x_conv,
                        l_prev_gru[i]: x_hid,
                        l_boxes[i]: x_boxes,
                        l_input_img2[i]: x_img2,
                    }, deterministic=True)
                else:
                    output[i] = lasagne.layers.get_output([l_out[i], l_gru[i], l_gru[i]], {
                        l_cell_input[i]: x_rnn,
                        l_input_reg[i]: x_conv,
                        l_prev_gru[i]: x_hid,
                    }, deterministic=True)
        elif CFG[i]['MODE'] == 'transformer':
            if CFG[i]['VISUALIZE'] > 0:
                output[i] = lasagne.layers.get_output([l_out[i], l_gru[i], l_loc[i], l_decoder[i]], {  # ,l_input_loc,l_loc1],{#,l_sel_region2], {#l_loc], {
                    l_cell_input[i]: x_rnn,
                    l_input_reg[i]: x_conv,
                    l_prev_gru[i]: x_hid,
                }, deterministic=True)

                if CFG[i]['TRANS_FEEDBACK']:
                    output[i] = lasagne.layers.get_output([l_out[i], l_gru[i], l_weighted_region[i]], {
                        l_cell_input[i]: x_rnn,
                        l_input_reg[i]: x_conv,
                        l_prev_gru[i]: x_hid,
                        l_weighted_region_prev[i]: x_region,
                    }, deterministic=True)
                else:
                    output[i] = lasagne.layers.get_output([l_out[i], l_gru[i], l_gru[i]], {
                        l_cell_input[i]: x_rnn,
                        l_input_reg[i]: x_conv,
                        l_prev_gru[i]: x_hid,
                    }, deterministic=True)

        elif CFG[i]['MODE'] == 'tensor-feedback' or CFG[i]['MODE'] == 'tensor-feedback2':
            if CFG[i]['PROPOSALS'] in [3, 4]:
                output[i] = lasagne.layers.get_output([l_out[i], l_gru[i], l_region[i]], {
                    l_cell_input[i]: x_rnn,
                    l_input_reg[i]: x_conv,
                    l_prev_gru[i]: x_hid,
                    l_boxes[i]: x_boxes,
                    l_input_img2[i]: x_img2,
                    l_region_feedback[i]: x_region,
                }, deterministic=True)
            elif CFG[i]['PROPOSALS'] > 1:
                print(
                    "proposals versions other that 3 and 4 are not supported with feedback")
                sys.exit()
            else:
                output[i] = lasagne.layers.get_output([l_out[i], l_gru[i], l_region[i]], {
                    l_cell_input[i]: x_rnn,
                    l_input_reg[i]: x_conv,
                    # l_input_cell: x_cell,
                    l_prev_gru[i]: x_hid,
                    l_region_feedback[i]: x_region,
                }, deterministic=True)

        f[i] = theano.function([x_rnn, x_conv, x_hid, x_region,
                                x_boxes, x_img2], output[i], on_unused_input='warn')

        f_sent[i] = theano.function(
            [x_sentence_sym], emb_sent[i], on_unused_input='warn')
    return f_cnn, emb_cnn, f, f_sent
    #}}}


def predict_batch(CFG, word_to_index, index_to_word, f_cnn, emb_cnn, f, f_sent, x_cnn, x_conv, sample=False):
    # {{{1
    x_sentence = np.zeros(
        (CFG['BATCH_SIZE'], CFG['SEQUENCE_LENGTH']), dtype='int32')
    x_input = np.zeros((CFG['BATCH_SIZE'], 1), dtype='int32')
    x_hid = np.zeros(
        (CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), dtype='float32')
    x_cell = np.zeros(
        (CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), dtype='float32')
    x_region = np.zeros(
        (CFG['BATCH_SIZE'], CFG['NUM_REGIONS']), dtype='float32')
    sent = []
    for i in range(CFG['SEQUENCE_LENGTH']):
        if i == 0:
            x_rnn = f_cnn(x_cnn)[:, np.newaxis]
        else:
            x_rnn = f_sent(x_input)
        if CFG['MODE'] == 'tensor-feedback':
            p0, hid, x_region = f(x_rnn[:, 0, :], x_conv, x_hid, x_region)
        else:
            p0, hid, cell = f(x_rnn[:, 0, :], x_conv, x_hid, x_cell)
            x_cell = cell
        x_hid = hid
        if sample:
            pa = np.zeros((p0.shape[0], 1), dtype=np.int)
            for b in range(pa.shape[0]):
                pa[b] = np.random.choice(p0.shape[2], 1, p=p0[b, 0])
        else:
            pa = p0.argmax(2)
        if i != 0:  # it works also without this, but with this is better!!!!
            x_input[:, 0] = pa[:, 0]
        x_sentence[:, i] = pa[:, 0]  # ,0]#tok
        if 0:
            print ' '.join([index_to_word[x] for x in x_sentence[0, 1:]])
            raw_input()
    for batch_id in range(CFG['BATCH_SIZE']):
        words = []
        for x in x_sentence[batch_id, 1:]:
            w = index_to_word[x]
            if w == '#END#':
                break
            words.append(w)
        sent.append(' '.join(words))
    return sent
    #}}}


def predict_batch_vis(CFG, word_to_index, index_to_word, f_cnn, emb_cnn, f, f_sent, x_cnn, img_id, x_img, x_img2, x_boxes, x_conv, html, tex_file, num_imgs, sample=False):
    # {{{1
    tex_images = []
    tex_words = []
    tex_line = 0
    attention = {'boxes': [], 'prob': [], 'word': []}
    x_sentence = np.zeros(
        (CFG['BATCH_SIZE'], CFG['SEQUENCE_LENGTH']), dtype='int32')
    x_input = np.zeros((CFG['BATCH_SIZE'], 1), dtype='int32')
    x_hid = np.zeros(
        (CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), dtype='float32')
    if CFG['MODE'] == 'tensor-feedback' or CFG['MODE'] == 'tensor-feedback2':
        x_cell = np.zeros(
            (CFG['BATCH_SIZE'], CFG['NUM_REGIONS']), dtype='float32')
    elif CFG['MODE'] == 'transformer' and CFG['TRANS_FEEDBACK']:
        x_cell = np.zeros(
            (CFG['BATCH_SIZE'], CFG['REGION_SIZE']), dtype='float32')
    else:
        x_cell = np.zeros(
            (CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), dtype='float32')
    sent = []
    bpos = 0
    for i in range(CFG['SEQUENCE_LENGTH']):
        if i == 0:
            if CFG['CNN_FINE_TUNE']:
                if CFG['MODE'] == 'normal':
                    x_rnn = f_cnn(x_img)
                if CFG['PROPOSALS'] == 1:
                    x_rnn, x_conv = f_cnn(x_img, x_boxes)
                elif CFG['PROPOSALS'] > 1:
                    x_rnn, x_conv, conv_before_pooling = f_cnn(
                        x_img, x_img2, x_boxes)
                else:
                    x_rnn, x_conv = f_cnn(x_img)
            else:
                x_rnn = f_cnn(x_cnn)[:, np.newaxis]
        else:
            x_rnn = f_sent(x_input)[:, 0]
        if CFG['FORCE_GT_SENTENCE'] != None:
            if i == 0:
                skip_word = 0
                im_ann = CFG['FORCE_GT_SENTENCE']
                sentences = im_ann[img_id[0]]['sentence']
                gt_words = sentences[0].lower().split(' ')
            else:
                if i + skip_word - 1 == len(gt_words):
                    break
                if not word_to_index.has_key(gt_words[i + skip_word - 1]):
                    print "GT:", '#NAW#'
                    widx = word_to_index['#NAW#']
                else:
                    print "GT:", gt_words[i + skip_word - 1]
                    widx = word_to_index[gt_words[i + skip_word - 1]]
                win = np.array(widx, dtype=np.int32).reshape((1, 1))
                x_rnn = f_sent(win)[:, 0]
        if CFG['MODE'] == 'transformer':
            p0, hid, feedback_regions, loc, r0 = f(
                x_rnn, x_conv, x_hid, x_cell, x_boxes, x_img2)
            x_cell = feedback_regions
        else:
            p0, hid, r0 = f(x_rnn, x_conv, x_hid, x_cell, x_boxes, x_img2)
        x_hid[:] = hid
        if CFG['MODE'] == 'tensor-feedback' or CFG['MODE'] == 'tensor-feedback2':
            x_cell = r0
        if sample:
            pa = np.zeros((p0.shape[0], 1), dtype=np.int)
            for b in range(pa.shape[0]):
                pa[b] = np.random.choice(p0.shape[2], 1, p=p0[b, 0])
        else:
            pa = p0.argmax(2)
        if i != 0:  # it works also without this, but with this is better!!!!
            x_input[:, 0] = pa[:, 0]
        x_sentence[:, i] = pa[:, 0]  # ,0]#tok
        if 1 and i > 0:
            import pylab
            pylab.figure(1)
            pylab.clf()
            wrd = index_to_word[pa[:, 0][bpos]]
            print wrd
            if wrd == "#END#" and CFG['FORCE_GT_SENTENCE'] == None:
                break
            if x_img2.shape[-1] == 1:
                my_img = (x_img[bpos] + CNN.MEAN_VALUES)[::-
                                                         1].transpose((1, 2, 0))
                pylab.gray()
                pylab.imshow(my_img.astype(np.uint8))
                pylab.draw()
            else:
                my_img = (x_img2[bpos] + CNN.PIXEL_MEANS[:,
                                                         np.newaxis, np.newaxis])[::-1].transpose((1, 2, 0))
                pix_val = np.array(
                    [245.543396,  231.89300537,  205.96020508], np.float32)
                img_sy = np.where(my_img[:, 0, :] == pix_val)[0]
                if len(img_sy) > 0:
                    img_sy = img_sy[0]
                else:
                    img_sy = my_img.shape[0]
                img_sx = np.where(my_img[0, :, :] == pix_val)[0]
                if len(img_sx) > 0:
                    img_sx = img_sx[0]
                else:
                    img_sx = my_img.shape[1]
                my_img = my_img[:img_sy, :img_sx]
                pylab.imshow(my_img.astype(np.uint8))
                pylab.draw()
                # raw_input()

            ca = pylab.gca()
            ca.get_xaxis().set_visible(False)
            ca.get_yaxis().set_visible(False)

            if CFG['SAVE_IMG'] != '' and i == 1:
                im_name = img_id[0] + '_0' + '.jpg'
                pylab.savefig(CFG['SAVE_IMG'] + im_name,
                              bbox_inches='tight', pad_inches=0)
                html_str = """
                Image {} <br><img src="{}" height="300"><br> <br>
                """.format(im_name, im_name)
                html.write(html_str)
                if not CFG['SKIP_BASIC_IMG']:
                    tex_images.append(im_name)
                    tex_words.append(im_name.split('_')[0])

            if CFG['PROPOSALS'] == 5:
                pylab.hold(True)
                regions = r0.sum(3)[0, 0]
                print regions.max()
                if CFG.has_key('CONV_REDUCED') and CFG['CONV_REDUCED']:
                    att = regions.reshape((14, 14))
                else:
                    att = regions.reshape((28, 28))
                att = scipy.misc.imresize(
                    att, (my_img.shape[0:2]), interp='bilinear', mode=None)
                pylab.imshow(att, alpha=0.6)

            elif CFG['PROPOSALS'] > 1:
                save_boxes = np.zeros((CFG['NUM_REGIONS'], 2, 5))
                if CFG['VISUALIZE'] == 2:
                    prob_im = np.zeros((x_img2.shape[2:]))
                    my_boxes = np.round(
                        x_boxes[CFG['NUM_REGIONS'] * bpos:CFG['NUM_REGIONS'] * (bpos + 1), 1:]).astype(np.int)
                    for b in np.arange(len(my_boxes)):
                        prob_im[my_boxes[b, 1]:my_boxes[b, 3], my_boxes[b, 0]:my_boxes[b, 2]] += r0[0, 0, b].sum(
                        ) * np.ones((my_boxes[b, 3] - my_boxes[b, 1], my_boxes[b, 2] - my_boxes[b, 0]))
                    pylab.hold(True)
                    pylab.imshow(prob_im, alpha=0.7)
                elif CFG['VISUALIZE'] == 1:
                    my_boxes = np.round(
                        x_boxes[CFG['NUM_REGIONS'] * bpos:CFG['NUM_REGIONS'] * (bpos + 1), 1:]).astype(np.int)
                    for idb, b in enumerate(my_boxes):
                        pylab.plot([b[0], b[0], b[2], b[2], b[0]], [
                                   b[1], b[3], b[3], b[1], b[1]], 'k', lw=30 * r0[0, 0, idb].sum())
                        pylab.plot([b[0], b[0], b[2], b[2], b[0]], [
                                   b[1], b[3], b[3], b[1], b[1]], color=CFG['VIS_COLOR'], lw=27 * r0[0, 0, idb].sum())
                        save_boxes[idb] = [b[0], b[0], b[2], b[2], b[0]], [
                            b[1], b[3], b[3], b[1], b[1]]
                    pylab.xlim(0, img_sx)
                    pylab.ylim(img_sy, 0)
                    regions = r0[0, 0].sum(1)

            elif CFG['MODE'] == 'transformer':
                regions = r0.sum(3)[0, 0]
                if CFG['TRANS_MULTIPLE_BOXES']:
                    kx = 3 / 2.
                    ky = 3 / 2.
                    tx = 0.5
                    ty = 0.5
                    pix = 16
                    print regions.max()
                    save_boxes = np.zeros((len(loc), 2, 5))
                    for idb, b in enumerate(loc):
                        p = np.array([[-kx, -ky, 1], [-kx, ky, 1],
                                      [kx, ky, 1], [kx, -ky, 1], [-kx, -ky, 1]])
                        A = b.reshape((2, 3))
                        pt = np.dot(A, p.T) * pix
                        if CFG['CLIP_BOXES']:
                            pylab.plot(np.clip(pt[0] + tx * pix, 0, 224), np.clip(
                                pt[1] + ty * pix, 0, 224), 'k', lw=30 * (regions[idb]))
                            pylab.plot(np.clip(pt[0] + tx * pix, 0, 224), np.clip(
                                pt[1] + ty * pix, 0, 224), color=CFG['VIS_COLOR'], lw=27 * (regions[idb]))
                        else:
                            pylab.plot(pt[0] + tx * pix, pt[1] +
                                       ty * pix, 'k', lw=30 * regions[idb])
                            pylab.plot(pt[0] + tx * pix, pt[1] +
                                       ty * pix, 'r', lw=27 * regions[idb])
                            save_boxes[idb] = [
                                pt[0] + tx * pix, pt[1] + ty * pix]
                    if CFG['CLIP_BOXES']:
                        pylab.xlim((0, 224))
                        pylab.ylim((224, 0))
                else:
                    kx = 3
                    ky = 3
                    for b in loc:
                        p = np.array([[0, 0, 1], [0, ky, 1], [
                                     kx, ky, 1], [kx, 0, 1], [0, 0, 1]])
                        A = b.reshape((2, 3))
                        print A
                        pt = np.dot(A, p.T) * 32
                        pylab.plot(pt[0], pt[1], 'r', lw=3)
                pylab.draw()
                pylab.show()
                # raw_input()
                # continue

            else:
                pylab.hold(True)
                if CFG['MODE'] == 'tensor-feedback' or CFG['MODE'] == 'tensor-feedback2':
                    regions = r0[0]
                else:
                    regions = r0.sum(3)[0, 0]
                print regions.max()
                num_smp = len(np.arange(14)[::CFG['CONV_REDUCED']])
                att = regions.reshape((num_smp, num_smp))
                att = scipy.misc.imresize(
                    att, (my_img.shape[0:2]), interp='bilinear', mode=None)
                pylab.imshow(att, alpha=0.6)
                pylab.title(wrd)
                pylab.draw()
                if CFG['MODE'] == 'tensor-feedback' or CFG['MODE'] == 'tensor-feedback2':
                    pass
                else:
                    pylab.figure(3)
                    pylab.clf()
                    pylab.title('Conditioned on word')
                    regions = r0[0, 0, :, pa[:, 0][0]]
                    print regions.max() / p0.max(2)[0]
                    pylab.imshow(regions.reshape((num_smp, num_smp)))

            ca = pylab.gca()
            ca.get_xaxis().set_visible(False)
            ca.get_yaxis().set_visible(False)
            pylab.draw()
            pylab.show()
            if CFG['SAVE_IMG'] != '':
                im_name = img_id[0] + '_%d' % i + '.jpg'
                pylab.savefig(CFG['SAVE_IMG'] + im_name,
                              bbox_inches='tight', pad_inches=0)
                html_str = """
                <img src="{}" height="200">
                """.format(im_name)
                html.write(html_str)
                tex_images.append(im_name)
                tex_words.append(wrd)
            elif CFG['SAVE_ATT'] != '':
                attention['boxes'] = save_boxes
                attention['prob'].append(regions)
                attention['word'].append(wrd)
                # raw_input()
            else:
                import pdb
                pdb.set_trace()

            # raw_input()
        if 0:
            print ' '.join([index_to_word[x] for x in x_sentence[0, 1:]])
            import pdb
            pdb.set_trace()
    if CFG['SAVE_IMG'] != '':
        html_str = "<br>"
        html.write(html_str)
        if num_imgs == -1:
            num_imgs == len(tex_images)
        tex_file.write(tex.begin_fig(num_imgs))
        # tex_file.write('\\small\r')
        for r in range(int(np.ceil((float(len(tex_images)) / float(num_imgs))))):
            tex_file.write(tex.images(tex_images[num_imgs * r:num_imgs * (
                r + 1)], tex_words[num_imgs * r:num_imgs * (r + 1)], num_imgs))
            tex_line += 1
        tex_file.write(tex.end_fig())
    if CFG['SAVE_ATT'] != '':
        import cPickle as pickle
        att_name = img_id[0] + '.pkl'
        # Save the attention file
        att_file = open(CFG['SAVE_ATT'] + att_name, 'wb')
        pickle.dump(attention, att_file)
        att_file.close()

    for batch_id in range(CFG['BATCH_SIZE']):
        words = []
        for x in x_sentence[batch_id, 1:]:
            w = index_to_word[x]
            if w == '#END#':
                break
            words.append(w)
        sent.append(' '.join(words))
    return sent
    # }}}


def evaluate_batch(CFG, word_to_index, index_to_word, f_cnn, emb_cnn, f, f_sent, x_cnn, x_img, x_img2, x_boxes, x_conv, sentence, sample=False, quiet_eval=False):
    # {{{

    x_sentence = np.zeros(
        (CFG['BATCH_SIZE'], CFG['SEQUENCE_LENGTH']), dtype='int32')
    x_input = np.zeros((CFG['BATCH_SIZE'], 1), dtype='int32')
    x_hid = np.zeros(
        (CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), dtype='float32')

    if CFG['MODE'] == 'tensor-feedback' or CFG['MODE'] == 'tensor-feedback2':
        x_cell = np.zeros(
            (CFG['BATCH_SIZE'], CFG['NUM_REGIONS']), dtype='float32')
    elif CFG['MODE'] == 'transformer' and CFG['TRANS_FEEDBACK']:
        x_cell = np.zeros(
            (CFG['BATCH_SIZE'], CFG['REGION_SIZE']), dtype='float32')
    else:
        x_cell = np.zeros(
            (CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), dtype='float32')
    sent = []
    bpos = 0
    pa = np.zeros((20, 1))
    probas = np.zeros((20, 1))
    for i in range(CFG['SEQUENCE_LENGTH']):
        if i == 0:
            if CFG['CNN_FINE_TUNE']:
                if CFG['MODE'] == 'normal':
                    x_rnn = f_cnn(x_img)
                if CFG['PROPOSALS'] == 1:
                    x_rnn, x_conv = f_cnn(x_img, x_boxes)
                elif CFG['PROPOSALS'] > 1:
                    x_rnn, x_conv, conv_before_pooling = f_cnn(
                        x_img, x_img2, x_boxes)
                else:
                    x_rnn, x_conv = f_cnn(x_img)
            else:
                x_rnn = f_cnn(x_cnn)[:, np.newaxis]
        else:
            x_rnn = f_sent(x_input)[:, 0]
        if CFG['MODE'] == 'transformer':
            p0, hid, cell = f(x_rnn, x_conv, x_hid, x_cell, x_boxes, x_img2)
        else:
            p0, hid, r0 = f(x_rnn, x_conv, x_hid, x_cell, x_boxes, x_img2)
        x_hid[:] = hid
        if CFG['MODE'] == 'tensor-feedback' or CFG['MODE'] == 'tensor-feedback2':
            x_cell = r0
        # Get the probability of the word that we do have.
        nb_sent = len(p0[:, 0, 0][:])

        for sent_index in range(nb_sent):
            split_sent = sentence[sent_index].split(" ")
            if i < len(split_sent):
                # Evaluate the probability of the current chosen word.
                try:
                    word_index = word_to_index[split_sent[i]]
                    print(split_sent[i])
                except Exception as e:
                    print("There was some problem")
                    print(e)
                    import pdb
                    pdb.set_trace()  # XXX BREAKPOINT

                loc_prob = p0[sent_index, 0, word_index]
                if i == 0:
                    print("supposed to store: ", loc_prob)
                    print("as index", sent_index)
                    probas[sent_index][0] = loc_prob
                    print("What we actually stored:")
                    print(probas[sent_index][0])
                    if loc_prob == 0:
                        print("found a local prob of 0")
                        import pdb
                        pdb.set_trace()  # XXX BREAKPOINT
                    if probas[sent_index][0] == 0:
                        print("Something is awry with zero probs")
                        import pdb
                        pdb.set_trace()  # XXX BREAKPOINT

                else:
                    old_proba = probas[sent_index][0]
                    probas[sent_index][0] = loc_prob * probas[sent_index][0]
                    if probas[sent_index][0] == 0:
                        print("found a zero: ", probas[sent_index][0])
                        print("At index: ", sent_index)
                        import pdb
                        pdb.set_trace()  # XXX BREAKPOINT

                    # accept the word coming from the sentence
                pa[sent_index][0] = word_index

                if i != 0:  # it works also without this, but with this is better!!!!
                    x_input[:, 0] = pa[:, 0]

                x_sentence[:, i] = pa[:, 0]

                if (not quiet_eval) and i > 0:
                    wrd = index_to_word[pa[:, 0][bpos]]
                    print("Iteration: ", i)
                    print("split_sent length: ", len(split_sent))
                    print(split_sent)
                    print wrd
                    if wrd == "#END#":
                        break

    for batch_id in range(CFG['BATCH_SIZE']):
        words = []
        last_w = '#END#'
        for x in x_sentence[batch_id, 0:]:
            w = index_to_word[x]
            if w == '#END#' or w == '#START#':
                break
            if last_w == w:
                # if index_to_word[x]
                continue
            words.append(w)
            last_w = w
        sent.append(' '.join(words))

    return sent, probas
    # }}}


def predict_batch_beam(CFG, word_to_index, index_to_word, f_cnn, emb_cnn, f, f_sent, x_cnn, x_img, x_img2, x_boxes, x_conv, beam_size=5):
    # {{{
    x_sentence = np.zeros(
        (beam_size, CFG['BATCH_SIZE'], CFG['SEQUENCE_LENGTH']), dtype='int32')
    x_back = np.zeros(
        (beam_size, CFG['BATCH_SIZE'], CFG['SEQUENCE_LENGTH']), dtype='int32')
    x_input = np.zeros((beam_size, CFG['BATCH_SIZE'], 1), dtype='int32')
    x_output = np.zeros(
        (beam_size, beam_size, CFG['BATCH_SIZE'], 1), dtype='int32')
    x_hid = np.zeros(
        (beam_size, CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), dtype='float32')
    x_hid_out = np.zeros(
        (beam_size, CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), dtype='float32')
    if CFG['MODE'] == 'tensor-feedback' or CFG['MODE'] == 'tensor-feedback2':
        x_cell = np.zeros(
            (beam_size, CFG['BATCH_SIZE'], CFG['NUM_REGIONS']), dtype='float32')
        x_cell_out = np.zeros(
            (beam_size, CFG['BATCH_SIZE'], CFG['NUM_REGIONS']), dtype='float32')
    elif CFG['MODE'] == 'transformer' and CFG['TRANS_FEEDBACK']:
        x_cell = np.zeros(
            (beam_size, CFG['BATCH_SIZE'], CFG['REGION_SIZE']), dtype='float32')
        x_cell_out = np.zeros(
            (beam_size, CFG['BATCH_SIZE'], CFG['REGION_SIZE']), dtype='float32')
    else:
        x_cell = np.zeros(
            (beam_size, CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), dtype='float32')
        x_cell_out = np.zeros(
            (beam_size, CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), dtype='float32')
    scr = np.zeros((beam_size, beam_size, CFG['BATCH_SIZE']))  # (past,present)
    # (past,present)
    cum_scr = np.zeros((beam_size, CFG['BATCH_SIZE'], CFG['SEQUENCE_LENGTH']))
    score = np.zeros(CFG['BATCH_SIZE'])
    finished = np.zeros((beam_size, CFG['BATCH_SIZE']), dtype=np.bool)
    # i=0
    if CFG['CNN_FINE_TUNE']:
        if CFG['MODE'] == 'normal':
            x_rnn = f_cnn(x_img)
        else:
            if CFG['PROPOSALS'] == 1:
                x_rnn, x_conv = f_cnn(x_img, x_boxes)
            elif CFG['PROPOSALS'] > 1:
                x_rnn, x_conv, _ = f_cnn(x_img, x_img2, x_boxes)
            else:
                x_rnn, x_conv = f_cnn(x_img)
    else:
        x_rnn = f_cnn(x_cnn)
    p0, hid, cell = f(x_rnn, x_conv, x_hid[0], x_cell[0], x_boxes, x_img2)
    x_hid[:] = hid
    x_cell[:] = cell
    pa = p0.argpartition(-beam_size, 2)[:, 0, -beam_size:]
    idx = np.tile(np.arange(p0.shape[0]), [beam_size, 1])
    cum_scr[:, :, 0] = p0[idx, 0, pa.T]
    # generated beam_size hyotheses
    for i in range(1, CFG['SEQUENCE_LENGTH']):
        for beam in range(beam_size):
            x_rnn = f_sent(x_input[beam])
            p0, hid, cell = f(x_rnn[:, 0, :], x_conv,
                              x_hid[beam], x_cell[beam], x_boxes, x_img2)
            x_hid_out[beam, :, :] = hid  # [:,0,:]
            x_cell_out[beam, :, :] = cell  # [:,0,:]
            pa = p0.argpartition(-beam_size, 2)[:, 0, -beam_size:]
            aux = p0[idx, 0, pa.T]
            # stop the update of a finished sentence
            aux[:, finished[beam]] = 1
            scr[beam, :, :] = aux * cum_scr[beam, :, i - 1]
            x_output[beam, ..., 0] = pa.T
        flatscr = scr.reshape((-1, CFG['BATCH_SIZE']))
        best_scr = np.argsort(-flatscr, 0)[:beam_size]
        cum_scr[:, :, i] = flatscr[best_scr, idx]
        x_back[:, :, i] = best_scr / beam_size
        x_input[:, :, 0] = x_output.reshape(
            (-1, CFG['BATCH_SIZE']))[best_scr, idx]
        x_hid = x_hid_out.reshape(
            (-1, CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']))[best_scr / beam_size, idx]
        x_cell = x_cell_out.reshape(
            (-1, CFG['BATCH_SIZE'], x_cell.shape[2]))[best_scr / beam_size, idx]
        x_sentence[:, :, i] = x_input[:, :, 0]
        finished = np.logical_or(
            finished, x_sentence[:, :, i] == word_to_index['#END#'])
        # print finished
        if 0:
            print "Scr", scr[:, :, 0].flatten()
            print "Words", [index_to_word[x] for x in x_output[:, :, 0].flatten()]
            print "Best scr", best_scr[:, 0]
            print "Cum scr", cum_scr[:, 0, i]
            print "Back scr", x_back[:, 0, i]
            print "Sentence", [index_to_word[x] for x in x_sentence[:, 0, i]]
            import pdb
            pdb.set_trace()
        if 0:
            print ' '.join([index_to_word[x] for x in x_sentence[0, 1:]])
            import pdb
            pdb.set_trace()
    # backward reconstruction
    sent = []
    prob = []
    for batch_id in range(CFG['BATCH_SIZE']):
        words = []
        words2 = []
        score[batch_id] = cum_scr[:, batch_id, -1].max()
        sel = cum_scr[:, batch_id, -1].argmax()
        prob.append(cum_scr[sel, batch_id, -1])
        words.append(index_to_word[x_sentence[sel, :, -1][0]])
        for p in range(CFG['SEQUENCE_LENGTH'])[-2:0:-1]:
            sel = x_back[sel, batch_id, p + 1]
            w = index_to_word[x_sentence[sel, batch_id, p]]
            if 0:  # batch_id==0:
                print [index_to_word[x_sentence[ff, batch_id, p]] for ff in range(beam_size)]
                print sel
                import pdb
                pdb.set_trace()
            words.append(w)
        for idp, p in enumerate(words[::-1]):
            if p == '#END#':
                break
            words2.append(p)
        sent.append(' '.join(words2))
    return sent, prob
    # }}}


def predict_batch_beam_ensemble(CFG, word_to_index, index_to_word, f_cnn, emb_cnn, f, f_sent, x_cnn, x_img, x_img2, x_boxes, x_conv, beam_size=5):
    # {{{

    x_output = dict()
    x_hid = dict()
    x_hid_out = dict()
    x_cell = dict()
    x_cell_out = dict()
    x_rnn = dict()
    x_conv_input = x_conv  # Backwards compatibility
    x_conv = dict()

    x_sentence = np.zeros(
        (beam_size, CFG[1]['BATCH_SIZE'], CFG[1]['SEQUENCE_LENGTH']), dtype='int32')
    x_input = np.zeros((beam_size, CFG[1]['BATCH_SIZE'], 1), dtype='int32')
    x_back = np.zeros(
        (beam_size, CFG[1]['BATCH_SIZE'], CFG[1]['SEQUENCE_LENGTH']), dtype='int32')
    x_output = np.zeros(
        (beam_size, beam_size, CFG[1]['BATCH_SIZE'], 1), dtype='int32')
    for i in range(1, len(CFG) + 1):
        x_hid[i] = np.zeros((beam_size, CFG[i]['BATCH_SIZE'],
                             CFG[i]['EMBEDDING_SIZE']), dtype='float32')
        x_hid_out[i] = np.zeros(
            (beam_size, CFG[i]['BATCH_SIZE'], CFG[i]['EMBEDDING_SIZE']), dtype='float32')

        # CFG['NUM_REGIONS'] different?? OO
        if CFG[i]['MODE'] == 'tensor-feedback' or CFG[i]['MODE'] == 'tensor-feedback2':
            x_cell[i] = np.zeros(
                (beam_size, CFG[i]['BATCH_SIZE'], CFG[i]['NUM_REGIONS']), dtype='float32')
            x_cell_out[i] = np.zeros(
                (beam_size, CFG[i]['BATCH_SIZE'], CFG[i]['NUM_REGIONS']), dtype='float32')
        else:
            x_cell[i] = np.zeros(
                (beam_size, CFG[i]['BATCH_SIZE'], CFG[i]['EMBEDDING_SIZE']), dtype='float32')
            x_cell_out[i] = np.zeros(
                (beam_size, CFG[i]['BATCH_SIZE'], CFG[i]['EMBEDDING_SIZE']), dtype='float32')

    # (past,present)
    scr = np.zeros((beam_size, beam_size, CFG[1]['BATCH_SIZE']))
    cum_scr = np.zeros(
        (beam_size, CFG[1]['BATCH_SIZE'], CFG[1]['SEQUENCE_LENGTH']))  # (past,present)
    score = np.zeros(CFG[1]['BATCH_SIZE'])
    finished = np.zeros((beam_size, CFG[1]['BATCH_SIZE']), dtype=np.bool)

    # i=0
    for j in range(1, len(CFG) + 1):
        if CFG[j]['CNN_FINE_TUNE']:
            if CFG[j]['MODE'] == 'normal':
                x_rnn[j] = f_cnn[j](x_img)
            else:
                if CFG[j]['PROPOSALS'] == 1:
                    x_rnn[j], x_conv[j] = f_cnn[j](x_img, x_boxes)
                elif CFG[j]['PROPOSALS'] > 1:
                    x_rnn[j], x_conv[j], _ = f_cnn[j](x_img, x_img2, x_boxes)
                else:
                    x_rnn[j], x_conv[j] = f_cnn[j](x_img)
        else:
            x_rnn[j] = f_cnn[j](x_cnn[j])
            print("You should not be here, this will very probably break")
            import pdb
            pdb.set_trace()
            x_conv[j] = x_conv_input

    p0 = dict()
    hid = dict()
    cell = dict()
    for j in range(1, len(CFG) + 1):
        p0[j], hid[j], cell[j] = f[j](x_rnn[j], x_conv[j], x_hid[j][0], x_cell[j][0],
                                      x_boxes, x_img2)
        x_hid[j][:] = hid[j]
        x_cell[j][:] = cell[j]

    # Compute an arithmetical average of the different distributions.
    total_number = 1
    total_p0 = p0[1]
    for j in range(2, len(CFG) + 1):
        total_p0 = total_p0 + p0[j]
        total_number = total_number + 1
    total_p0 = total_p0 / total_number

    pa = total_p0.argpartition(-beam_size, 2)[:, 0, -beam_size:]
    idx = np.tile(np.arange(total_p0.shape[0]), [beam_size, 1])
    cum_scr[:, :, 0] = total_p0[idx, 0, pa.T]

    for i in range(1, CFG[1]['SEQUENCE_LENGTH']):
        for beam in range(beam_size):
            for j in range(1, len(CFG) + 1):  # Loop over the models.
                x_rnn[j] = f_sent[j](x_input[beam])
                p0[j], hid[j], cell[j] = f[j](
                    x_rnn[j][:, 0, :], x_conv[j], x_hid[j][beam], x_cell[j][beam], x_boxes, x_img2)
                x_hid_out[j][beam, :, :] = hid[j]  # [:,0,:]
                x_cell_out[j][beam, :, :] = cell[j]  # [:,0,:]

            # Average all the p0 produced:
            total_number = 1
            total_p0 = p0[1]
            for j in range(2, len(CFG) + 1):
                total_p0 = total_p0 + p0[j]
                total_number = total_number + 1
            total_p0 = total_p0 / total_number

            # Keep going...
            pa = total_p0.argpartition(-beam_size, 2)[:, 0, -beam_size:]
            aux = total_p0[idx, 0, pa.T]
            # stop the update of a finished sentence
            aux[:, finished[beam]] = 1
            scr[beam, :, :] = aux * cum_scr[beam, :, i - 1]
            x_output[beam, ..., 0] = pa.T

        flatscr = scr.reshape((-1, CFG[1]['BATCH_SIZE']))
        best_scr = np.argsort(-flatscr, 0)[:beam_size]
        cum_scr[:, :, i] = flatscr[best_scr, idx]
        x_back[:, :, i] = best_scr / beam_size

        x_input[:, :, 0] = x_output.reshape(
            (-1, CFG[1]['BATCH_SIZE']))[best_scr, idx]

        for j in range(1, len(CFG) + 1):
            x_hid[j] = x_hid_out[j].reshape(
                (-1, CFG[j]['BATCH_SIZE'], CFG[j]['EMBEDDING_SIZE']))[best_scr / beam_size, idx]
            x_cell[j] = x_cell_out[j].reshape(
                (-1, CFG[j]['BATCH_SIZE'], x_cell[j].shape[2]))[best_scr / beam_size, idx]

        x_sentence[:, :, i] = x_input[:, :, 0]
        finished = np.logical_or(
            finished, x_sentence[:, :, i] == word_to_index['#END#'])
        # print finished
        if 0:
            print "Scr", scr[:, :, 0].flatten()
            print "Words", [index_to_word[x] for x in x_output[:, :, 0].flatten()]
            print "Best scr", best_scr[:, 0]
            print "Cum scr", cum_scr[:, 0, i]
            print "Back scr", x_back[:, 0, i]
            print "Sentence", [index_to_word[x] for x in x_sentence[:, 0, i]]
            import pdb
            pdb.set_trace()
        if 0:
            print ' '.join([index_to_word[x] for x in x_sentence[0, 1:]])
            import pdb
            pdb.set_trace()
    # backward reconstruction
    sent = []
    prob = []
    for batch_id in range(CFG[1]['BATCH_SIZE']):
        words = []
        words2 = []
        score[batch_id] = cum_scr[:, batch_id, -1].max()
        sel = cum_scr[:, batch_id, -1].argmax()
        prob.append(cum_scr[sel, batch_id, -1])
        words.append(index_to_word[x_sentence[sel, :, -1][0]])
        for p in range(CFG[1]['SEQUENCE_LENGTH'])[-2:0:-1]:
            sel = x_back[sel, batch_id, p + 1]
            w = index_to_word[x_sentence[sel, batch_id, p]]
            if 0:
                print [index_to_word[x_sentence[ff, batch_id, p]] for ff in range(beam_size)]
                print sel
                import pdb
                pdb.set_trace()
            words.append(w)
        for idp, p in enumerate(words[::-1]):
            if p == '#END#':
                break
            words2.append(p)
        sent.append(' '.join(words2))
    return sent, prob
    # }}}


def predict_batch_beam_ensemble_v2(CFG, word_to_index, index_to_word, f_cnn, emb_cnn, f, f_sent, x_cnn, x_img, x_img2, x_boxes, x_conv, beam_size=5, quiet_eval=False):
    # {{{
    """
    New strategy for ensembling many models.
    """
    sent, prob = dict(), dict()
    all_sent_prob = dict()
    for i in range(1, 1 + len(f_cnn)):
        loc_f_cnn = f_cnn[i]
        loc_emb_cnn = emb_cnn[i]
        loc_f = f[i]
        loc_f_sent = f_sent[i]
        loc_x_cnn = x_cnn[i]
        loc_x_conv = x_conv[i]
        sent[i], prob[i] = predict_batch_beam(CFG[i], word_to_index, index_to_word, loc_f_cnn,
                                              loc_emb_cnn, loc_f, loc_f_sent, loc_x_cnn, x_img, x_img2, x_boxes, loc_x_conv, beam_size=5)

        nb_sent = len(sent[i])
        nb_model = len(f_cnn)
        print("Went there once")
        averaged_prob = np.zeros(nb_sent)

        for j in range(1, 1 + len(f_cnn)):
            # if not j==i: # No need to reevaluate
            # TODO maybe evaluate it as sum of log probs for better
            # stability.
            sentences, eval_prob = evaluate_batch(CFG[j], word_to_index, index_to_word, f_cnn[j], emb_cnn[j],
                                                  f[j], f_sent[j], x_cnn[j], x_img, x_img2, x_boxes, x_conv[i], sent[i], quiet_eval=True)
            if not (sentences == sent[i]):
                print("The sentence was somehow modified")
                # import pdb; pdb.set_trace()  # XXX BREAKPOINT
            else:
                averaged_prob += (1.0 / len(f_cnn)) * eval_prob[:, 0]
        all_sent_prob[i] = (sentences, averaged_prob)

    final_chosen_sentences = dict()
    final_probs = dict()
    for b in range(CFG[1]['BATCH_SIZE']):
        prob = -1
        for m in range(1, 1 + len(all_sent_prob)):
            new_prob = all_sent_prob[m][1][b]
            if new_prob > prob:
                prob = new_prob
                best_index = m
        final_chosen_sentences[b] = all_sent_prob[m][0][b]
        final_probs[b] = all_sent_prob[m][1][b]

    nb_sentences_no_double = int(len(final_chosen_sentences) / 2)
    final_chosen_sentences_no_doubles = dict()
    final_probs_no_doubles = dict()
    for i in range(nb_sentences_no_double):
        p1 = final_probs[2 * i]
        p2 = final_probs[2 * i + 1]
        if p1 > p2:
            final_chosen_sentences_no_doubles[i] = final_chosen_sentences[2 * i]
            final_probs_no_doubles[i] = final_probs[2 * i]
        else:
            final_chosen_sentences_no_doubles[i] = final_chosen_sentences[2 * i + 1]
            final_probs_no_doubles[i] = final_probs[2 * i + 1]
    return final_chosen_sentences_no_doubles, final_probs_no_doubles
    # }}}


def generateCaptionsExternalImages(imgdir, dbval, word_to_index, index_to_word, CFG, params, f_cnn, emb_cnn, f, f_sent, VISUALIZE, BEAM_SIZE, show_sent=True):
    # {{{
    blobs = []
    num_sent = 0
    import glob
    import pylab
    lstimg = glob.glob(imgdir + '/*.jpg')
    html = None
    tex_file = None
    num_imgs = 1
    import CNN

    if CFG['PROPOSALS'] == 1:
        im_size = 224
    elif CFG['PROPOSALS'] > 1:
        im_size = CFG['IM_SIZE']
    else:
        im_size = -1  # tell the data generator to not prepare images for proposals

    if params['shuffle'] == 0:
        shuffle = False
    else:
        import random
        random.seed(params['shuffle'])
        shuffle = True  # False
    if CFG['VISUALIZE']:
        assert(CFG['BATCH_SIZE'] == 1)
    if CFG['USE_FLIP']:
        mybatch = CFG['BATCH_SIZE'] / 2
    else:
        mybatch = CFG['BATCH_SIZE']
    if CFG['PROPOSALS'] > 0:
        PYSOLR_PATH = '/home/thoth/tlucas/links/imcap/imcap/edge_boxes_with_python'
        import sys
        if not PYSOLR_PATH in sys.path:
            sys.path.append(PYSOLR_PATH)

    for idimg in range(0, len(lstimg), mybatch):
        img_id = []
        print
        x_img = floatX(np.zeros((CFG['BATCH_SIZE'], 3, 224, 224)))
        for imb in range(mybatch):
            if idimg + imb < len(lstimg):
                img = lstimg[idimg + imb]
            else:
                img = lstimg[idimg]
            im_sample = pylab.imread(img)
            if CFG['USE_FLIP']:
                x_img[2 * imb:2 * imb + 1] = CNN.prep_image(im_sample)
                x_img[2 * imb + 1:2 * imb +
                      2] = CNN.prep_image(im_sample)[:, :, :, ::-1]
            else:
                x_img[imb:imb + 1] = CNN.prep_image(im_sample)
            print img
            if idimg + imb < len(lstimg):
                # img_id.append(img.split('_')[-1].split('.')[0])
                # for flickr30k
                img_id.append(img.split('/')[-1].split('.')[0])
                if CFG['USE_FLIP']:
                    img_id.append(img.split('_')[-1].split('.')[0])
            else:
                img_id.append(-1)
                if CFG['USE_FLIP']:
                    img_id.append(-1)
        x_img2 = floatX(np.zeros((CFG['BATCH_SIZE'], 3, 1, 1)))
        x_conv = floatX(np.zeros((CFG['BATCH_SIZE'], 1, 1, 1)))
        x_cnn = floatX(np.zeros((CFG['BATCH_SIZE'], 1)))
        x_boxes = floatX(-np.ones((CFG['BATCH_SIZE'] * CFG['NUM_REGIONS'], 5)))
        if CFG['PROPOSALS'] != 0:
            # get edgeboxes
            import edge_boxes
            count = 0
            num_boxes = CFG['NUM_REGIONS']
            max_im_size = 1.5 * CFG['IM_SIZE']
            #boxes = edge_boxes.get_windows(['/home/lear/mpederso/links/imcap'+img[1:]])[0][:,:4]

            import pdb
            pdb.set_trace()
            boxes_test = edge_boxes.get_windows(
                ['/home/lear/mpederso/links/imcap' + img[1:]])
            boxes = boxes_test[0][:, :4]

            x_img2 = floatX(
                np.zeros((CFG['BATCH_SIZE'], 3, max_im_size, max_im_size)))
            x_img2[...] = CNN.PIXEL_MEANS[np.newaxis,
                                          :, np.newaxis, np.newaxis]
            # ,3,max_im_size,max_im_size)))
            x_img2[count] = CNN.PIXEL_MEANS[:, np.newaxis, np.newaxis]
            newimgs2, scale = CNN.prep_image_RCNN(
                im_sample, scale=CFG['IM_SIZE'], max_size=max_im_size)
            x_img2[count, :, :newimgs2.shape[1], :newimgs2.shape[2]] = newimgs2
            order_boxes = np.arange(len(boxes))
            scaled_boxes = boxes[order_boxes] * scale
            dedup_boxes = scaled_boxes[:num_boxes]
            num_boxes = min(num_boxes, len(dedup_boxes))
            x_boxes[:, :num_boxes] = np.concatenate((count * np.ones((num_boxes, 1), dtype='int32'),
                                                     dedup_boxes[:, 1:2], dedup_boxes[:, 0:1], dedup_boxes[:, 3:4], dedup_boxes[:, 2:3]), 1)
            # print"Erorr, this script works only without proposals!!!"
            # sys.exit()
        if VISUALIZE:
            sent = predict_batch_vis(CFG, word_to_index, index_to_word, f_cnn, emb_cnn, f,
                                     f_sent, x_cnn, img_id, x_img, x_img2, x_boxes, x_conv, html, tex_file, num_imgs)
        else:
            tm = time.time()
            sent, prob = predict_batch_beam(CFG, word_to_index, index_to_word, f_cnn, emb_cnn,
                                            f, f_sent, x_cnn, x_img, x_img2, x_boxes, x_conv, beam_size=BEAM_SIZE)
            print('Time for ', CFG['BATCH_SIZE'], 'samples:', time.time() - tm)
            #sent = predict_batch(x_cnn,x_conv)
        if CFG['USE_ROTATIONS']:
            print"Not implemented for external images"
            sys.exit()
            # prob=prob.reshape((len(sent),2))
            blob_batch = []
            for ss in range(0, len(sent), 5):
                add = np.argmax(prob[ss:ss + 5])
                print(add)
                blob_batch.append(
                    {"image_id": int(img_id[ss]), "caption": sent[ss + add]})
        elif CFG['USE_FLIP']:
            # prob=prob.reshape((len(sent),2))
            if CFG['USE_ROTATIONS']:
                print('Not implemented yet!')
                sys.exit()
            blob_batch = []
            for ss in range(0, len(sent), 2):
                if prob[ss] > prob[ss + 1]:
                    add = 0
                else:
                    add = 1
                blob_batch.append(
                    {"image_id": int(img_id[ss]), "caption": sent[ss + add]})
        else:
            blob_batch = [
                {"image_id": int(img_id[x]), "caption":sent[x]} for x in range(len(sent))]
        blobs += blob_batch
        if show_sent:
            print num_sent, sent  # ,prob
        num_sent += mybatch
        if num_sent > params['test_length']:
            break
        if 0:
            import pylab
            pylab.figure(1)
            pylab.clf()
            print sent[0]
            im = img
            pylab.imshow(im)
            pylab.show()
            raw_input()
    import json
    savefile = CFG['RESULTS_EXTERNAL']  # 'captions.json'
    json.dump(blobs, open(savefile, "w"))
    # }}}


def generateCaptions(savefile, dbval, word_to_index, index_to_word, CFG, params, f_cnn, emb_cnn, f, f_sent, VISUALIZE, BEAM_SIZE, show_sent=True, ensemble=False):
    # {{{
    blobs = []
    num_sent = 0

    if ensemble:
        CFG_dict = CFG
        CFG = CFG_dict[1]

    if CFG['PROPOSALS'] == 1:
        im_size = 224
    elif CFG['PROPOSALS'] > 1:
        im_size = CFG['IM_SIZE']
    else:
        im_size = -1  # tell the data generator to not prepare images for proposals

    if params['shuffle'] == 0:
        shuffle = False
    else:
        import random
        random.seed(params['shuffle'])
        shuffle = True  # False
    if CFG['VISUALIZE']:
        assert(CFG['BATCH_SIZE'] == 1)
    if CFG['SAVE_IMG'] != '':
        html = open(CFG['SAVE_IMG'] + '/results.html', 'w')
        tex_file = open(CFG['SAVE_IMG'] + '/results.tex', 'w')
        tex_file.write(tex.begin_doc())
    else:
        html = None
        tex_file = None

    count = 0
    max_count = 50

    for img_id, x_img, x_img2, _, _, _, x_boxes in batch_gen(CFG, dbval, CFG['BATCH_SIZE'], im_size, word_to_index, shuffle=shuffle, force_missing_boxes=True, use_flip=CFG['USE_FLIP'], use_rotations=CFG['USE_ROTATIONS'], start_from=params['eval_from']):

        count += CFG['BATCH_SIZE']
        x_conv = floatX(np.zeros((CFG['BATCH_SIZE'], 1, 1, 1)))
        x_cnn = floatX(np.zeros((CFG['BATCH_SIZE'], 1)))
        if not ensemble:
            if VISUALIZE:
                sent = predict_batch_vis(CFG, word_to_index, index_to_word, f_cnn, emb_cnn, f, f_sent,
                                         x_cnn, img_id, x_img, x_img2, x_boxes, x_conv, html, tex_file, num_imgs=20)
            else:
                import time
                tm = time.time()
                sent, prob = predict_batch_beam(CFG, word_to_index, index_to_word, f_cnn, emb_cnn,
                                                f, f_sent, x_cnn, x_img, x_img2, x_boxes, x_conv, beam_size=BEAM_SIZE)
                print('Time for ', CFG['BATCH_SIZE'],
                      'samples:', time.time() - tm)

            if CFG['USE_ROTATIONS']:
                blob_batch = []
                for ss in range(0, len(sent), 5):
                    add = np.argmax(prob[ss:ss + 5])
                    print(add)
                    blob_batch.append(
                        {"image_id": int(img_id[ss]), "caption": sent[ss + add]})
            elif CFG['USE_FLIP']:
                if CFG['USE_ROTATIONS']:
                    print('Not implemented yet!')
                    sys.exit()
                blob_batch = []
                for ss in range(0, len(sent), 2):
                    if prob[ss] > prob[ss + 1]:
                        add = 0
                    else:
                        add = 1
                    blob_batch.append(
                        {"image_id": int(img_id[ss]), "caption": sent[ss + add]})
            else:
                blob_batch = [
                    {"image_id": int(img_id[x]), "caption":sent[x]} for x in range(len(sent))]
        else:
            if VISUALIZE:
                print(
                    "visualize option is not yet supported for an ensemble of architectures, sorry see you!")
                sys.exit()
            else:

                img_id_remapped = None
                # Set to one if you want to use the old ensembling strategy.
                if params['ensemble_strategy'] == 1:
                    sent, prob = predict_batch_beam_ensemble(
                        CFG_dict, word_to_index, index_to_word, f_cnn, emb_cnn, f, f_sent, x_cnn, x_img, x_img2, x_boxes, x_conv, beam_size=BEAM_SIZE)
                    sent = [sent[i] for i in range(len(sent)) if i % 2 == 0]
                    prob = [prob[i] for i in range(len(prob)) if i % 2 == 0]
                    img_id_remapped = [img_id[i]
                                       for i in range(len(img_id)) if i % 2 == 0]
                else:
                    sent, prob = predict_batch_beam_ensemble_v2(
                        CFG_dict, word_to_index, index_to_word, f_cnn, emb_cnn, f, f_sent, x_cnn, x_img, x_img2, x_boxes, x_conv, beam_size=BEAM_SIZE, quiet_eval=True)
                    img_id_remapped = [img_id[i]
                                       for i in range(len(img_id)) if i % 2 == 0]

            if not img_id_remapped is None:
                img_id = img_id_remapped
                max_num_sent = int(params['test_length'])
            else:
                max_num_sent = params['test_length']

            blob_batch = [{"image_id": int(img_id[x]), "caption":sent[x]} for x in range(
                len(sent)) if int(img_id[x]) != -1]

        blobs += blob_batch
        if show_sent:
            print num_sent, sent  # ,prob
        if CFG['USE_ROTATIONS']:
            num_sent += CFG['BATCH_SIZE'] / 5
        elif CFG['USE_FLIP']:
            num_sent += CFG['BATCH_SIZE'] / 2
        else:
            num_sent += CFG['BATCH_SIZE']
        try:
            assert(max_num_sent > 0)
        except:
            max_num_sent = params['test_length']

        if num_sent >= max_num_sent:
            break
        if 0:
            import pylab
            pylab.figure(1)
            pylab.clf()
            print sent[0]
            im = pylab.imread(
                '/home/lear/mpederso/links/myneuraltalk/data/coco/images/val2014/COCO_val2014_000000%06d.jpg' % int(batch[0][3]))
            pylab.imshow(im)
            pylab.show()
            raw_input()

    if CFG['SAVE_IMG'] != '':
        tex_file.write(tex.end_doc())
        tex_file.close()
    # write the json file
    json.dump(blobs, open(savefile, "w"))
    if CFG['RESULTS_EXTERNAL'] != 'captions.json':
        json.dump(blobs, open(CFG['RESULTS_EXTERNAL'], "w"))
    # }}}


def evaluateCaptions(captionfile):
    # {{{
    # Evaluation
    import sys
    sys.path.insert(
        0, '/scratch/algorab/tlucas/save_new_imcap/newpull_imccap/imcap/coco-caption')
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    import matplotlib.pyplot as plt
    import skimage.io as io
    import pylab
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)

    import json
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    dataDir = '.'
    dataType = 'val2014'
    algName = 'fakecap'
    annFile = '%s/coco/annotations/captions_%s.json' % (dataDir, dataType)
    resFile = captionfile  # 'sentT.json'

    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    cocoEval.evaluate()
    res = cocoEval.eval
    return res
    # }}}


def evaluateCaptionsFlickr(captionfile):
    # {{{
    # Evaluation
    import sys
    sys.path.insert(
        0, '/scratch/algorab/tlucas/save_new_imcap/newpull_imccap/imcap/coco-caption')

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    import matplotlib.pyplot as plt
    import skimage.io as io
    import pylab
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)

    import json
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    annFile = './flick_data/val_annot.json'
    resFile = captionfile  # 'sentT.json'

    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    cocoEval.evaluate()
    res = cocoEval.eval
    return res
    # }}}


from lasagne.utils import floatX
import CNN

from RNNTraining import batch_gen, load_coco


def cmd_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', default='',
                        help='pkl file with the trained model')
    parser.add_argument('--dataset', type=str,
                        default='coco', help='flickr or coco')
    parser.add_argument('-b', '--beam_size', dest='beam_size', type=int,
                        default=1, help='Size of the beam in the beam search')
    parser.add_argument('--ensemble_strategy', type=int, default=1,
                        help='1: first ensemble strategy (models averaged at every step) 2: Second strategy (models averaged later).')

    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=10, help='Size of the batch, default is 10')
    parser.add_argument('--conv_reduced', type=int, default=-1,
                        help='Reduces the size of the convolutional features by defining the amount of stride to read the activation layer')
    parser.add_argument('--forbid_naw', action='store_true',
                        help='Forbid to generate #NAW#; it can generate nonsensical sentences, but it improves a bit the final results')
    parser.add_argument('--external_images', type=str, default='',
                        help='Generate captions for the images at the given folder')
    parser.add_argument('--tensor_tied', type=int, default=-1,
                        help='Force a value for tensor tied independently of what used for training')
    parser.add_argument('-v', '--visualize', dest='visualize', type=int, default=0,
                        help='Visualize the attention for each word starting from sample x')
    parser.add_argument('-l', '--length', dest='test_length', type=int, default=5000,
                        help='Number of samples to evaluate (default 5000), but 1000 is also a good trade-off')
    parser.add_argument('--max_sent', dest='max_sent', type=int,
                        default=-1, help='Maximum number of words in a sentence')
    parser.add_argument('--num_proposals', dest='num_proposals', type=int, default=-1,
                        help='Nomber of proposals to use. If not set use the same number as in training')
    parser.add_argument('--im_size', type=int, default=-1,
                        help='Size of the image used for computing proposals')
    parser.add_argument('--shuffle', dest='shuffle', type=int, default=0,
                        help='Way test data is shuffled; 0:No shuffle >0 : use the given number as random seed initialization')
    parser.add_argument('--trans_stride', type=int, default=-1,
                        help='Set the value to use for the spatial transformer porposals stride. If not set uses the one used at training time')
    parser.add_argument('--save_img', type=str, default='',
                        help='Save an image per frame with attention in the give directory')
    parser.add_argument('--save_att', type=str, default='',
                        help='Save the attention per frame in the given directory')
    parser.add_argument('--force_gt_sentence', type=str,
                        default='', help='Force to generate the gt sentences')
    parser.add_argument('--save_img_columns', type=int, default=13,
                        help='Number of columns used for saving the images')
    parser.add_argument('--vis_color', type=str, default='r',
                        help='Color used to draw the boxes')
    parser.add_argument('--results_external', type=str, default='captions.json',
                        help='Where to save the captioninng of external images')

    parser.add_argument('--force_proposals', action='store_true',
                        help='Force to use proposals even if it has been trained with grid')
    parser.add_argument('--force_transformer', action='store_true',
                        help='Force to use the spatial transformer even if it has been trained with grid')
    parser.add_argument('--clip_boxes', action='store_true',
                        help='Clip the boxes generated by the spatial transoform network. Only for visualization.')
    parser.add_argument('--add_validation', action='store_true',
                        help='Force the evaluation to be on the 10K validation subset even if the training was done only on training data (useful to evaluate exactly on the same data)')
    parser.add_argument('--use_test_split', action='store_true',
                        help='Evaluate on 5k test split')
    parser.add_argument('--use_newtest_split', action='store_true',
                        help='Evaluate on 5k test split, disjoint also from the additional training data!!!!')
    parser.add_argument('--ensemble', type=int, default=-1,
                        help='If greater than 1, evaluates with an ensemble of models.')
    parser.add_argument('--ensemble_id', type=int, default=-1,
                        help='Only for flickr: Which set of .pkl weights to load.')
    parser.add_argument('--V2', type=int, default=-1,
                        help='if set to true, will use new strategy')
    parser.add_argument('--use_flip', action='store_true',
                        help='Evaluate also with flipped images and keep the best sentence!')
    parser.add_argument('--use_rotations', action='store_true',
                        help='Evaluate also with 4 addtional rotated images and keep the best sentence!')
    parser.add_argument('--trans_force_norot', action='store_true',
                        help='Force the spartial transformer to not use rotations even if it has been trained with rotations')
    parser.add_argument('--force_all_validation', action='store_true',
                        help='Force to use all validation data even though it has been used for training. Needed to submit to the coco server')
    parser.add_argument('--eval_from', type=int, default=0,
                        help='Start the evaluation from the nth image')
    parser.add_argument('--skip_basic_img', action='store_true',
                        help='Skip the first image without boxes.')
    parser.add_argument('--karpathy_split', action='store_true',
                        help='Use Karpathy split for validation and test.')
    parser.add_argument('--relax_check_init', type=int, default=0,
                        help='Check init will still warn, but no longer call raw_input. You should not be using this unless running --dissect.')

    return parser


if __name__ == "__main__":
    # parse command line
    parser = cmd_line_parser()

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print(params['filename'])
    print(params['ensemble'])

    # This just means that we don't want to pass the path as a parameter.
    if params['filename'] == 'HardEnsemble':

        if params['dataset'] == 'coco':
            first_model_path = '/home/thoth/tlucas/image_attention/new_imcap/imcap/nresults/vgg/vgg_tranformer_feedback_finetuned_alldataflip_batch10/transformer_10.pkl'
        elif params['dataset'] == 'flickr':
            assert(params['ensemble_id'] != -
                   1), "You did not give the ID, and with flickr I want you to do it."

            from ensemble_paths import ensemble_flick_1, ensemble_flick_2, ensemble_flick_3, ensemble_flick_4, ensemble_flick_5, ensemble_flick_6
            if params['ensemble_id'] in range(1, 7):
                ensembles_list = [ensemble_flick_1, ensemble_flick_2, ensemble_flick_3,
                                  ensemble_flick_4, ensemble_flick_5, ensemble_flick_6]
                ensemble_flick = ensembles_list[params['ensemble_id']]
            else:
                print("For now, ensemble id " +
                      str(params['ensemble_id']) + " is not supported")
                import pdb
                pdb.set_trace()

            first_model_path = ensemble_flick[0]
            ensemble_paths = ensemble_flick[1:]

        d = pickle.load(open(first_model_path))
    else:
        d = pickle.load(open(params['filename']))

    ###remove #NAW#
    if params['forbid_naw']:
        d['param values']['l_tensor.b_hw'][-1] = -100
        d['param values']['l_tensor.b_rw'][-1] = -100
    ###remove #NAW#
    vocab = d['vocab']
    word_to_index = d['word_to_index']
    index_to_word = d['index_to_word']
    CFG = d['config']
    CFG['ENSEMBLE'] = params['ensemble']
    # overwrite some parameters

    if params['external_images'] != '':
        pass  # params['visualize']=1

    net = buildNetwork(CFG, params, vocab)
    setParamNetwork(CFG, net['out'], net['cnn'],
                    net['sent_emb'], net['out_reg'], d)

    if CFG.has_key('ENSEMBLE') and CFG['ENSEMBLE'] > 0:
        #from ensemble_paths import ensemble_coco_1 as ensemble_paths

        ds, CFGs, nets = [d], [CFG], [net]
        for model_path in ensemble_paths:
            print("Building one model..."),
            # Load d (param values, dict, config)
            ds.append(pickle.load(open(model_path)))
            CFGs.append(ds[-1]['config'])  # Net config.
            nets.append(buildNetwork(CFGs[-1], params, vocab))
            setParamNetwork(CFGs[-1], nets[-1]['out'], nets[-1]['cnn'],
                            nets[-1]['sent_emb'], nets[-1]['out_reg'], ds[-1])
            print("Ok!!")

        CFG_dict, net_dict = {}, {}
        for i in range(len(CFGs)):
            # This is pretty ugly, I know..
            CFG_dict[i + 1] = CFGs[i]
            net_dict[i + 1] = nets[i]

        f_cnn, emb_cnn, f, f_sent = compileEnsemble(CFG_dict, net_dict)

    else:
        f_cnn, emb_cnn, f, f_sent = compileNetwork(CFG, net)

    if params['dataset'] == "coco":
        if params['karpathy_split']:  # dbval,
            dbval, dbtest = load_coco(no_train=False, karpathy_split=True)
        elif params['force_all_validation']:
            CFG['ADD_VALIDATION'] = False
            dbtrain, dbval = load_coco(
                no_train=False, add_validation=CFG['ADD_VALIDATION'], new_test=CFG['USE_NEWTEST_SPLIT'])
        else:
            dbval, dbtest = load_coco(
                no_train=True, add_validation=CFG['ADD_VALIDATION'], new_test=CFG['USE_NEWTEST_SPLIT'])
    elif params['dataset'] == "flickr":
        _, dbval, dbtest = load_flickr(no_train=True, no_test=False)

    print('Validation length:', len(dbval))
    # raw_input()

    if CFG['USE_TEST_SPLIT'] and not params['karpathy_split']:
        dbval = dbtest

    if params['external_images'] != '':
        generateCaptionsExternalImages(params['external_images'], dbval, word_to_index, index_to_word, CFG,
                                       params, f_cnn, emb_cnn, f, f_sent, VISUALIZE=params['visualize'], BEAM_SIZE=params['beam_size'])
    else:
        import os
        if not os.path.exists('./temp'):
            os.makedirs('./temp')
        import uuid
        captionfile = './temp/' + str(uuid.uuid4()) + ".json"

        if CFG.has_key('ENSEMBLE') and CFG['ENSEMBLE'] > 0:
            ensemble = True
            generateCaptions(captionfile, dbval, word_to_index, index_to_word, CFG_dict, params, f_cnn, emb_cnn,
                             f, f_sent, VISUALIZE=params['visualize'], BEAM_SIZE=params['beam_size'], ensemble=ensemble)
        else:
            ensemble = False
            generateCaptions(captionfile, dbval, word_to_index, index_to_word, CFG, params, f_cnn, emb_cnn,
                             f, f_sent, VISUALIZE=params['visualize'], BEAM_SIZE=params['beam_size'], ensemble=ensemble)

        if params['dataset'] == 'coco':
            res = evaluateCaptions(captionfile)
        elif params['dataset'] == 'flickr':
            res = evaluateCaptionsFlickr(captionfile)
        os.remove(captionfile)
        # }}}
