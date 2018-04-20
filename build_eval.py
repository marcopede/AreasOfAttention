# coding: utf-8
# vim:fdm=marker
import argparse
import numpy as np
import theano
from lasagne.utils import floatX
import lasagne
import sys
import os
import scipy
import theano.tensor as T
import matplotlib
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
import CNN

from RNNTraining import build_loc_net
from flickr_prepareData import load_flickr



def get_CFG(CFG, params):
    # {{{
    """
    Update CFG using params.
    """
    #-----------------------------------------------------------
    #                      Building CFG
    #-----------------------------------------------------------

    # Those parameters are copied directly from params.
    cfg_from_param = ['skip_basic_img', 'results_external', 'trans_force_norot',
                      'vis_color', 'save_img_columns', 'clip_boxes', 'use_rotations',
                      'use_test_split', 'use_newtest_split', 'visualize', 'save_img',
                      'save_att', 'dataset']
    CFG.update({key.upper(): params[key] for key in cfg_from_param})

    # If those parameters are not present in CFG, set them to false.
    default_to_false = ['TRANS_ADD_BIG_PROPOSALS', 'TRANS_LOCNET_DROPOUT', 'TRANS_REGLEARN',
                        'TRANS_NOLEARN', 'TRANS_SLOWLEARN', 'TRANS_FEEDBACK',
                        'TRANS_NOROT', 'TRANS_ZEROPAD', 'TENSOR_ADD_CONV', 'CLEAN_MASKS',
                        'TRANS_MULTIPLE_BOXES', 'TRANS_USE_STATE']
    for k in default_to_false:
        CFG[k] = False if k not in CFG.keys() else CFG[k]

    # If those parameters are not present, give them default values.
    cfg_default_values = {'TRANS_ZOOM': 1.0, 'TRANS_STRIDE': 1, 'TRANS_LOCNET': 0,
                          'TENSOR_TIED': 1, 'FEEDBACK': 2, 'IM_SIZE': 224,
                          'CNN_FINE_TUNE': False, 'IM_SIZE': 400, 'START_NORMALIZED': 0}
    CFG.update({key: cfg_default_values[key] for key in cfg_default_values.keys() if key not in CFG.keys()})

    CFG['FORCE_GT_SENTENCE'] = pickle.load(open('flickr30k/annotations.dict')) if (params['force_gt_sentence'] != '') else None
    CFG['USE_FLIP'] = 'both' if params['use_flip'] else None
    assert not (CFG['USE_ROTATIONS'] and (CFG['BATCH_SIZE'] % 5 != 0)), 'Error, to evaluate with rotations you need to use a batch that is a multiple of 5'
    assert not (CFG['USE_FLIP'] and CFG['BATCH_SIZE'] % 2 != 0), 'Error, to evaluate with flips you need to use a batch that is a multiple of 2'

    if params['trans_stride'] != -1:
        CFG['TRANS_STRIDE'] = params['trans_stride']
    if (params['conv_reduced'] != -1):
        CFG['CONV_REDUCED'] = params['conv_reduced']
    if params['force_transformer']:
        CFG.update({'MODE': 'transformer', 'FORCE_TRANSFORMER': True, 'TRANS_MULTIPLE_BOXES': True, 'TRANS_USE_PRETRAINED': True,
                    'TRANS_ZEROPAD': 1, 'TRANS_STRIDE': CFG['CONV_REDUCED']})
    else:
        CFG['FORCE_TRANSFORMER'] = False

    if params['force_proposals']:
        CFG['PROPOSALS'], CFG['FORCE_PROPOSALS'] = 4, True
    else:
        CFG['FORCE_PROPOSALS'] = False

    # Should we add (a subset of) the validation data?
    if not CFG.has_key("ADD_VALIDATION"):
        CFG["ADD_VALIDATION"] = False
    if params['add_validation']:
        CFG["ADD_VALIDATION"] = True
    if params['num_proposals'] != -1:
        CFG['NUM_REGIONS'] = params['num_proposals']
    if params['max_sent'] != -1:
        CFG['SEQUENCE_LENGTH'] = params['max_sent']

    CFG['BATCH_SIZE'] = 1 if CFG['VISUALIZE'] > 0 else params['batch_size']

    if (params['save_img'] != '') and not os.path.exists(params['save_img']):
        os.makedirs(params['save_img'])

    if (params['save_att'] != '') and (not os.path.exists(params['save_att'])):
        os.makedirs(params['save_att'])

    # not use anymore precomputed CNN features
    CFG['CNN_FINE_TUNE'] = True

    # Remove one of the three matrices from the three-way interaction.
    if not CFG.has_key('DISSECT'):
        CFG['DISSECT'] = 'No'
    # CFG['DISSECT']=params['dissect']

    #CFG['IMGFEEDBACK_MECHANISM'] = params['imgfeedback_mechanism']

    if params['tensor_tied'] != -1:
        CFG['TENSOR_TIED'] = params['tensor_tied']

    if params['im_size'] != -1:
        CFG['IM_SIZE'] = params['im_size']

    if CFG['MODE'] == 'transformer':
        CFG['NUM_REGIONS'] = (
            (14 + CFG['TRANS_STRIDE'] - 1) / CFG['TRANS_STRIDE'])**2

    elif CFG['PROPOSALS'] == 0:
        CFG['NUM_REGIONS'] = (
            (14 + CFG['CONV_REDUCED'] - 1) / CFG['CONV_REDUCED'])**2

    if 'CNN_MODEL' not in CFG.keys():
        # Retrocompatibility with weights trained prior to resnet inclusion
        CFG['CNN_MODEL'] = "vgg"

    return CFG
    # }}}


def build_CNN(CFG):
    # {{{
    """
    """
    if CFG['CNN_MODEL'] == "vgg":
        import CNN
        vgg16 = CNN.build_model()
        print "Loading pretrained VGG16 parameters"
        model_param_values = pickle.load(open('vgg16.pkl'))['param values']
        lasagne.layers.set_all_param_values(
            vgg16['prob'], model_param_values)
        l_input_img = vgg16['input']
        l_input_cnn = vgg16['fc7_dropout']
        #l_input_regions = vgg16['conv5_3']
        cnn = vgg16
    elif CFG["CNN_MODEL"] == "resnet":
        import resnet_CNN
        resnet50 = resnet_CNN.build_model(
            input_layer=None, batch_size=CFG['BATCH_SIZE'])
        model_param_values = pickle.load(open('resnet50.pkl'))[
            'param values']

        from save_layers import add_names_layers_and_params, set_param_dict
        add_names_layers_and_params(resnet50)
        #lasagne.layers.set_all_param_values(resnet50['prob'], model_param_values)
        set_param_dict(resnet50['prob'], model_param_values,
                       prefix='', show_layers=False, relax=False)
        l_input_img = resnet50['input']

        if CFG['RESNET_LAYER'] == 'prob':
            # ExpressionLayer(vgg16['fc7_dropout'],lambda X:X[:,0,:], output_shape='auto')#the first bbox is the entire image
            l_input_cnn = resnet50['prob']
        elif CFG['RESNET_LAYER'] == 'pool5':
            l_input_cnn = resnet50['pool5']
        else:
            print("Layer not supported for resnet")
            raise ValueError()
            #print(bcolors.FAIL + "Resnet not yet handled" + bcolors.ENDC)
        cnn = resnet50
    else:
        print(bcolors.FAIL + "Unknown CNN selected" + bcolors.ENDC)

    return cnn, l_input_img, l_input_cnn
    # }}}

def build_finetune_proposals(CFG, vgg16, resnet50):
    # {{{
    # use images at different resolution but without fully connected layers
    assert((vgg16 is None) or (resnet50 is None)), "Only one cnn can be used"
    _input_regions, l_out_reg, l_input_img2, l_boxes, l_input_regions, l_conv = 6 * [None]

    if CFG['PROPOSALS'] == 3:
        vgg16_det = CNN.build_model_RCNN( CFG['NUM_REGIONS'], CFG['IM_SIZE'] * 1.5, pool_dims=3, dropout_value=CFG['RNN_DROPOUT'])
        print "Loading pretrained VGG16 parameters for detection"
        l_conv = vgg16_det['conv5_3']
        l_input_img2 = vgg16_det['input']
        l_boxes = vgg16_det['boxes']
        l_input_regions = vgg16_det['reshape']

        if CFG['CONV_NORMALIZED'] == 1:
            l_input_regions = ExpressionLayer(l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
        elif CFG['CONV_NORMALIZED'] == 2:
            l_input_regions = ExpressionLayer(l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
        else:
            _input_regions = ExpressionLayer(l_input_regions, lambda X: X * 0.01, output_shape='auto')

        l_cnn_embedding2 = DenseLayer(l_input_regions, num_units=CFG['REGION_SIZE'], name='l_cnn_proposals')
        l_input_regions = ReshapeLayer(
            l_cnn_embedding2, (CFG['BATCH_SIZE'], CFG['NUM_REGIONS'], CFG['REGION_SIZE'], 1))

        l_input_regions = lasagne.layers.DimshuffleLayer(l_input_regions, (0, 2, 1, 3))
        l_out_reg = l_input_regions
        l_input_reg = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS'], 1), name='l_input_reg')
        l_input_regions = ReshapeLayer(l_input_reg, (CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regions')

    # use images at different resolution but without fully connected layers
    elif CFG['PROPOSALS'] == 4:

        vgg16_det = CNN.build_model_RCNN(CFG['NUM_REGIONS'], int(CFG['IM_SIZE'] * 1.5), pool_dims=1, dropout_value=CFG['RNN_DROPOUT'])

        print "Loading pretrained VGG16 parameters for detection"
        model_param_values = pickle.load(open('vgg16.pkl'))['param values']
        lasagne.layers.set_all_param_values(vgg16_det['conv5_3'], model_param_values[:-6])
        l_input_img2 = vgg16_det['input']
        l_conv = vgg16_det['conv5_3']
        l_boxes = vgg16_det['boxes']
        l_input_regions = vgg16_det['crop']
        l_input_regions = ReshapeLayer(l_input_regions, (CFG['BATCH_SIZE'] * CFG['NUM_REGIONS'], CFG['REGION_SIZE']))

        if CFG['CONV_NORMALIZED'] == 1:
            l_input_regions = ExpressionLayer(l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
        elif CFG['CONV_NORMALIZED'] == 2:
            l_input_regions = ExpressionLayer(l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
        else:
            _input_regions = ExpressionLayer(l_input_regions, lambda X: X * 0.01, output_shape='auto')
        l_input_regions = ReshapeLayer(
            l_input_regions, (CFG['BATCH_SIZE'], CFG['NUM_REGIONS'], CFG['REGION_SIZE'], 1))
        l_input_regions = lasagne.layers.DimshuffleLayer(
            l_input_regions, (0, 2, 1, 3))
        l_out_reg = l_input_regions
        l_input_reg = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS'], 1), name='l_input_reg')
        l_input_regions = ReshapeLayer(
            l_input_reg, (CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regions')

    # use images at different resolution but without fully connected layers
    elif CFG['PROPOSALS'] == 5:
        vgg16_det = CNN.build_model_RCNN(CFG['NUM_REGIONS'], int(
            CFG['IM_SIZE'] * 1.5), pool_dims=1, dropout_value=CFG['RNN_DROPOUT'])
        print "Loading pretrained VGG16 parameters for detection"
        model_param_values = pickle.load(open('vgg16.pkl'))[
            'param values']
        lasagne.layers.set_all_param_values(
            vgg16_det['conv5_3'], model_param_values[:-6])
        l_input_img2 = vgg16_det['input']
        l_conv = vgg16_det['conv5_3']
        l_input_regions = vgg16_det['conv5_3']
        if CFG['CONV_REDUCED'] > 1:
            l_input_regions = ExpressionLayer(l_input_regions, lambda X: X[:, :, ::CFG['CONV_REDUCED'], ::CFG['CONV_REDUCED']], output_shape='auto')

        if CFG['CONV_NORMALIZED'] == 1:
            l_input_regions = ExpressionLayer(l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
        elif CFG['CONV_NORMALIZED'] == 2:
            l_input_regions = ExpressionLayer(l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
        else:
            _input_regions = ExpressionLayer(l_input_regions, lambda X: X * 0.01, output_shape='auto')
        l_out_reg = l_input_regions
        l_input_reg = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS'], 1), name='l_input_reg')
        l_input_regions = ReshapeLayer(l_input_reg, (CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regions')

    else:
        if CFG['CNN_MODEL'] == 'vgg':
            l_out_reg = vgg16['conv5_3']
        elif CFG['CNN_MODEL'] == 'resnet':
            l_out_reg = resnet50['res4f_relu']
        else:
            print(bcolors.FAIL + "Unrecognized network" + bcolors.ENDC)

        l_input_reg = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE'], 14, 14), name='l_input_reg')
        if CFG['CONV_REDUCED']:
            # added a scaling factor of 100 to avoid exploding gradients
            l_input_regions = ExpressionLayer(l_input_reg, lambda X: X[:, :, ::CFG['CONV_REDUCED'], ::CFG['CONV_REDUCED']], output_shape='auto')
        else:
            l_input_regions = l_input_reg
        if CFG['CONV_NORMALIZED'] == 1:
            l_input_regions = ExpressionLayer(l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
        elif CFG['CONV_NORMALIZED'] == 2:
            l_input_regions = ExpressionLayer(l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
        else:
            l_input_regions = ExpressionLayer(l_input_regions, lambda X: X * 0.01, output_shape='auto')
        if CFG['TENSOR_ADD_CONV']:
            l_input_regions = ConvLayer(l_input_regions, num_filters=CFG['REGION_SIZE'], filter_size=(
                3, 3), pad='same', name='l_add_con')
        l_input_regions = ReshapeLayer( l_input_regions, (CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS'], 1))

    return l_input_regions, _input_regions, l_out_reg, l_input_img2, l_boxes, l_conv
    # }}}


def buildNetwork(CFG, params, vocab):
    # {{{
    """
    TODO document me
    """

    # Use params to update CFG
    CFG = get_CFG(CFG, params)

    #-----------------------------------------------------------
    #           Setting up the image Embedding.
    #-----------------------------------------------------------
    l_input_sentence = InputLayer((CFG['BATCH_SIZE'], 1), name='l_input_sentence')  # input (1 word)
    l_sentence_embedding = lasagne.layers.EmbeddingLayer(l_input_sentence,
                                                         input_size=len(vocab),
                                                         output_size=CFG['EMBEDDING_SIZE'],
                                                         name='l_sentence_embedding')

    # Setting up CNN in case of fine tuning.
    if CFG['CNN_FINE_TUNE']:
        cnn, l_input_cnn, l_input_img = build_CNN(CFG)

        if CFG['CNN_MODEL'] == "vgg":
            vgg16, resnet50 = cnn, None
        elif CFG["CNN_MODEL"] == "resnet":
            vgg16, resnet50 = None, cnn

        if CFG['START_NORMALIZED'] == 1:
            l_input_cnn = ExpressionLayer(l_input_cnn, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
        elif CFG['START_NORMALIZED'] == 2:
            l_input_cnn = ExpressionLayer(l_input_cnn, lambda X: X / T.sqrt(
                T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
    else:
        l_input_cnn = InputLayer((CFG['BATCH_SIZE'], CFG['CNN_FEATURE_SIZE']), name='l_input_cnn')

    l_cnn_embedding = DenseLayer(l_input_cnn, num_units=CFG['EMBEDDING_SIZE'],
                                 nonlinearity=lasagne.nonlinearities.identity, name='l_cnn_embedding')

    l_cnn_embedding2 = ReshapeLayer(l_cnn_embedding, ([0], 1, [1]), name='l_cnn_embedding2')

    l_rnn_input = InputLayer((CFG['BATCH_SIZE'], 1, CFG['EMBEDDING_SIZE']), name='l_rnn_input')
    l_dropout_input = DropoutLayer(
        l_rnn_input, p=0.5, name='l_dropout_input')

    l_input_reg = None
    l_out_reg = None
    l_decoder = None
    l_region_feedback = None
    l_region = None
    l_input_img2 = None
    l_boxes = None
    l_conv = None
    l_loc = None
    l_loc1 = None
    l_input_loc = None
    l_sel_region2 = None
    l_weighted_region_prev = None
    l_weighted_region = None

    input_shape = (CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE'])
    if CFG['MODE'] == 'normal':
        # {{{1
        l_cell_input = InputLayer(input_shape, name='l_cell_input')
        l_prev_gru = InputLayer(input_shape, name="l_prev_gru")
        l_gru = GRUMemoryLayer(CFG['EMBEDDING_SIZE'],
                               l_cell_input, l_prev_gru, name='l_gru')

        l_dropout_output = DropoutLayer(
            l_gru, p=0.5, name='l_dropout_output')
        # decoder is a fully connected layer with one output unit for each word in the vocabulary
        l_decoder = DenseLayer(l_dropout_output, num_units=len(
            vocab), nonlinearity=lasagne.nonlinearities.softmax, name='l_decoder')

        l_out = ReshapeLayer(
            l_decoder, ([0], 1, [1]), name='l_out')
        # }}}

    elif CFG['MODE'] == 'tensor':
        l_cell_input = InputLayer(input_shape, name='l_cell_input')
        l_prev_gru = InputLayer(input_shape, name="l_prev_gru")
        l_gru = GRUMemoryLayer(CFG['EMBEDDING_SIZE'], l_cell_input, l_prev_gru, name='l_gru')
        l_dropout_output = DropoutLayer(l_gru, p=0.5, name='l_dropout_output')
        l_dropout_output = ReshapeLayer(l_dropout_output, ([0], 1, [1]), name='l_dropout_output')

            # TODO put me back

        if CFG['CNN_FINE_TUNE']:
            l_input_regions, _input_regions, l_out_reg, l_input_img2, l_boxes, l_conv = build_finetune_proposals(CFG, vgg16, resnet50)
        else:
            l_input_regions = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regions')

        # TODO a block.
        #l_decoder = build_decoderLayer(l_dropout_output, l_input_regions, vocab, CFG)
        if CFG.has_key('DISSECT') and CFG['DISSECT'] != 'No':
            if CFG['DISSECT'] == 'wr':
                l_decoder = TensorProdFactLayer((l_dropout_output, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'],
                                                dim_w=len(vocab), nonlinearity=softmax, name='l_tensor', W_hr='skip', b_hr='skip')
            elif CFG['DISSECT'] == 'rs':
                l_decoder = TensorProdFactLayer((l_dropout_output, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'],
                                                dim_w=len(vocab), nonlinearity=softmax, name='l_tensor', W_rw='skip', b_rw='skip')
        if CFG['DISSECT'] == 'wr':
            l_decoder = TensorProdFactLayer((l_dropout_output, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'],
                                            dim_w=len(vocab), nonlinearity=softmax, name='l_tensor', W_hr='skip', b_hr='skip')
        elif CFG['DISSECT'] == 'rs':
            l_decoder = TensorProdFactLayer((l_dropout_output, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'],
                                            dim_w=len(vocab), nonlinearity=softmax, name='l_tensor', W_rw='skip', b_rw='skip')
        else:
            l_decoder = TensorProdFactLayer((l_dropout_output, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'],
                                            dim_w=len(vocab), nonlinearity=softmax, name='l_tensor')

        l_out = ExpressionLayer(l_decoder, lambda X: X.sum(2), output_shape='auto', name='l_out')  # sum over regions

    elif CFG['MODE'] == 'transformer':
        #{{{2
        print(bcolors.OKGREEN + "Transformer mode." + bcolors.ENDC)
        from TProd3 import TensorProdFactLayer, WeightedSumLayer, SubsampleLayer

        # define a cell
        l_cell_input = InputLayer((CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), name='l_cell_input')
        from agentnet.memory import GRUMemoryLayer
        l_prev_gru = InputLayer((CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), name="l_prev_gru")

        if CFG['TRANS_FEEDBACK']:
            l_weighted_region_prev = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE']), name="l_weighted_region_prev")
            if CFG['FEEDBACK'] == 2:
                l_cell_concat = lasagne.layers.ConcatLayer(
                    [l_cell_input, l_weighted_region_prev], axis=1, name='l_cell_concat')
            else:
                print("Are you sure you don't want to use feedback=2? I think you should. Change your mind, then come to see me again.")
        else:
            l_cell_concat = l_cell_input

        l_gru = GRUMemoryLayer(CFG['EMBEDDING_SIZE'],
                               l_cell_concat, l_prev_gru, name='l_gru')
        l_dropout_output = DropoutLayer(l_gru, p=CFG['RNN_DROPOUT'], name='l_dropout_output')
        l_dropout_output = ReshapeLayer(l_dropout_output, ([0], 1, [1]), name='l_dropout_output')

        if CFG['TRANS_USE_PRETRAINED']:
            l_out_reg = vgg16['conv5_2']
            #l_out_reg2 = vgg16['conv5_3']
        else:
            l_out_reg = vgg16['conv5_3']
        l_input_reg = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE'], 14, 14), name='l_input_reg')
        l_input_regions = l_input_reg
        if CFG['TRANS_USE_PRETRAINED']:
            l_input_regions = l_input_regions
        else:
            if CFG['CONV_NORMALIZED'] == 1:
                l_input_regions = ExpressionLayer(l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
            elif CFG['CONV_NORMALIZED'] == 2:
                l_input_regions = ExpressionLayer(l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
            else:
                l_input_regions = ExpressionLayer(l_input_regions, lambda X: X * 0.01, output_shape='auto')

        factor = 2.0
        W = lasagne.init.Constant(0.0)
        b = lasagne.init.Constant(0.0)
        if CFG['TRANS_MULTIPLE_BOXES']:
            num_prop, l_loc = build_loc_net(CFG, l_gru, l_input_regions, 1, (
                14, 14), (3, 3), CFG['TRANS_STRIDE'], CFG['TRANS_ZOOM'], W, b, name='')
            if CFG['TRANS_ADD_BIG_PROPOSALS']:
                num_prop_big, l_loc_big = build_loc_net(CFG, l_gru, l_input_regions, 1, (
                    14, 14), (3, 3), CFG['TRANS_STRIDE'], CFG['TRANS_ZOOM'] * 2, W, b, name='_big')
                l_loc = ConcatLayer((l_loc, l_loc_big), axis=0)
                num_prop += num_prop_big
            l_sel_region2 = MultiTransformerLayer(l_input_regions, l_loc, kernel_size=(
                3, 3), zero_padding=CFG['TRANS_ZEROPAD'])  # 3x3
            if CFG['TRANS_USE_PRETRAINED']:
                Wvgg = vgg16['conv5_3'].W.reshape(
                    (CFG['REGION_SIZE'], CFG['REGION_SIZE'] * 3 * 3)).swapaxes(0, 1)
                bvgg = vgg16['conv5_3'].b
                l_sel_region = DenseLayer(
                    l_sel_region2, num_units=CFG['REGION_SIZE'], name='l_sel_region', W=Wvgg, b=bvgg)
                if CFG['CONV_NORMALIZED'] == 1:
                    l_sel_region = ExpressionLayer(l_sel_region, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                elif CFG['CONV_NORMALIZED'] == 2:
                    l_sel_region = ExpressionLayer(l_sel_region, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
                else:
                    l_sel_region = l_sel_region
            else:
                l_sel_region = DenseLayer(
                    l_sel_region, num_units=CFG['REGION_SIZE'], name='l_sel_region')

            l_sel_region = ReshapeLayer(
                l_sel_region, (CFG['BATCH_SIZE'], num_prop, CFG['REGION_SIZE']))
            l_sel_region = DimshuffleLayer(l_sel_region, (0, 2, 1))
            l_sel_region = ReshapeLayer(
                l_sel_region, (CFG['BATCH_SIZE'], CFG['REGION_SIZE'], num_prop))
        else:
            b = np.zeros((2, 3), dtype='float32')
            b[0, 0] = 2
            b[1, 1] = 2
            b = b.flatten()
            W = lasagne.init.Constant(0.0)
            l_input_loc = l_gru
            if CFG['TRANS_USE_STATE']:
                l_input_im = ConvLayer(l_input_regions, num_filters=512, filter_size=(
                    3, 3), pad='same', name='l_reduce_im1')
                l_input_im = lasagne.layers.MaxPool2DLayer(l_input_im, (2, 2))
                l_input_im = ConvLayer(l_input_im, num_filters=512, filter_size=(
                    3, 3), pad='same', name='l_reduce_im2')
                l_input_im = lasagne.layers.MaxPool2DLayer(l_input_im, (2, 2))
                l_input_im = ReshapeLayer(l_input_im, (CFG['BATCH_SIZE'], 512))
                l_input_loc = ConcatLayer((l_gru, l_input_im))
            l_loc1 = DenseLayer(
                l_input_loc, num_units=256, name='l_loc1')
            l_loc = DenseLayer(
                l_loc1, num_units=6, W=W, b=b, nonlinearity=None, name='l_loc2')
            l_sel_region = TransformerLayer(
                l_input_regions, l_loc, downsample_factor=2)
            l_sel_region = DenseLayer(
                l_sel_region, num_units=CFG['REGION_SIZE'], name='l_sel_region')
            l_sel_region = ReshapeLayer(
                l_sel_region, (CFG['BATCH_SIZE'], CFG['REGION_SIZE'], 1))

        l_decoder = TensorProdFactLayer((l_dropout_output, l_sel_region), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
            vocab), W=lasagne.init.Normal(std=0.001, mean=0.0), nonlinearity=lasagne.nonlinearities.softmax, name='l_tensor')

        if CFG['TRANS_FEEDBACK']:
            l_region = ExpressionLayer(l_decoder, lambda X: X.sum(3), output_shape='auto', name='l_region')  # sum over regions
            l_weighted_region = WeightedSumLayer([l_sel_region, l_region], name='l_weighted_region')
        l_out = ExpressionLayer(l_decoder, lambda X: X.sum(
            2), output_shape='auto', name='l_out')  # sum over regions
        #}}}

    elif CFG['MODE'] == 'tensor-feedback':
        # {{{2
        # define a cell
        l_cell_input = InputLayer((CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), name='l_cell_input')
        l_region_feedback = InputLayer((CFG['BATCH_SIZE'], CFG['NUM_REGIONS']), name='l_region_feedback')
        l_cell_concat = lasagne.layers.ConcatLayer(
            [l_cell_input, l_region_feedback], axis=1, name='l_cell_concat')
        from agentnet.memory import GRUMemoryLayer
        l_prev_gru = InputLayer((CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), name="l_prev_gru")
        l_gru = GRUMemoryLayer(CFG['EMBEDDING_SIZE'],
                               l_cell_concat, l_prev_gru, name='l_gru')

        l_dropout_output = DropoutLayer(
            l_gru, p=0.5, name='l_dropout_output')
        l_dropout_output = ReshapeLayer(
            l_dropout_output, ([0], 1, [1]), name='l_dropout_output')

        from TProd3 import TensorProdFactLayer
        l_input_regions = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regions')
        l_tensor = TensorProdFactLayer((l_dropout_output, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
            vocab), nonlinearity=lasagne.nonlinearities.softmax, name='l_tensor')
        l_region = ExpressionLayer(l_tensor, lambda X: X.sum(
            3), output_shape='auto', name='l_region')  # sum over
        l_region = ReshapeLayer(
            l_region, ([0], [2]), name='l_region')

        l_out = ExpressionLayer(l_tensor, lambda X: X.sum(
            2), output_shape='auto', name='l_out')  # sum over regions
        #}}}

    elif CFG['MODE'] == 'tensor-feedback2':
        # {{{2
        l_feedback = InputLayer((CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), name='l_feedback')
        l_prev_gru = InputLayer((CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), name="l_prev_gru")
        from TProd3 import TensorProdFactLayer, WeightedSumLayer
        if CFG['PROPOSALS'] == 3:  # use images at different resolution but without fully connected layers
            import CNN
            vgg16_det = CNN.build_model_RCNN(
                CFG['NUM_REGIONS'], CFG['IM_SIZE'] * 1.5, pool_dims=3, dropout_value=CFG['RNN_DROPOUT'])
            print "Loading pretrained VGG16 parameters for detection"
            l_input_img2 = vgg16_det['input']
            l_conv = vgg16_det['conv5_3']
            l_boxes = vgg16_det['boxes']
            l_input_regions = vgg16_det['reshape']
            if CFG['CONV_NORMALIZED'] == 1:
                l_input_regions = ExpressionLayer(
                    l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
            elif CFG['CONV_NORMALIZED'] == 2:
                l_input_regions = ExpressionLayer(
                    l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
            else:
                l_input_regions = ExpressionLayer(
                    l_input_regions, lambda X: X * 0.01, output_shape='auto')
            l_cnn_embedding2 = DenseLayer(
                l_input_regions, num_units=CFG['REGION_SIZE'], name='l_cnn_proposals')
            l_input_regions = ReshapeLayer(
                l_cnn_embedding2, (CFG['BATCH_SIZE'], CFG['NUM_REGIONS'], CFG['REGION_SIZE'], 1))
            l_input_regions = lasagne.layers.DimshuffleLayer(
                l_input_regions, (0, 2, 1, 3))
            l_out_reg = l_input_regions
            l_input_reg = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS'], 1), name='l_input_reg')
            l_input_regions = ReshapeLayer(
                l_input_reg, (CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regions')

        # use images at different resolution but without fully connected layers
        elif CFG['PROPOSALS'] == 4:
            if CFG['CNN_MODEL'] == 'vgg':
                import CNN
                vgg16_det = CNN.build_model_RCNN(CFG['NUM_REGIONS'], int(
                    CFG['IM_SIZE'] * 1.5), pool_dims=1, dropout_value=CFG['RNN_DROPOUT'])
                print "Loading pretrained VGG16 parameters for detection"
                model_param_values = pickle.load(open('vgg16.pkl'))[
                    'param values']
                lasagne.layers.set_all_param_values(
                    vgg16_det['conv5_3'], model_param_values[:-6])
                l_input_img2 = vgg16_det['input']
                l_conv = vgg16_det['conv5_3']
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
                #lasagne.layers.set_all_param_values(resnet50['prob'], model_param_values)
                set_param_dict(
                    resnet50_det['pool5'], model_param_values, prefix='', show_layers=False, relax=False)
                l_input_img2 = resnet50_det['input']
                l_conv = resnet50_det['res4f_relu']
                l_boxes = resnet50_det['boxes']
                l_input_regions = resnet50_det['crop']
                l_input_regions = ReshapeLayer(
                    l_input_regions, (CFG['BATCH_SIZE'] * CFG['NUM_REGIONS'], CFG['REGION_SIZE']))

            if CFG['CONV_NORMALIZED'] == 1:
                l_input_regions = ExpressionLayer(
                    l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
            elif CFG['CONV_NORMALIZED'] == 2:
                l_input_regions = ExpressionLayer(
                    l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
            else:
                _input_regions = ExpressionLayer(
                    l_input_regions, lambda X: X * 0.01, output_shape='auto')
            l_input_regions = ReshapeLayer(
                l_input_regions, (CFG['BATCH_SIZE'], CFG['NUM_REGIONS'], CFG['REGION_SIZE'], 1))
            l_input_regions = lasagne.layers.DimshuffleLayer(
                l_input_regions, (0, 2, 1, 3))
            l_out_reg = l_input_regions
            l_input_reg = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS'], 1), name='l_input_reg')
            l_input_regions = ReshapeLayer(
                l_input_reg, (CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regions')
        else:
            if CFG['CNN_MODEL'] == 'vgg':
                l_out_reg = vgg16['conv5_3']
            elif CFG['CNN_MODEL'] == 'resnet':
                l_out_reg = resnet50['res4f_relu']
            else:
                print(bcolors.FAIL + "Unrecognized network" + bcolors.ENDC)

            l_input_reg = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE'], 14, 14), name='l_input_reg')
            if CFG['CONV_REDUCED'] > 1:
                # added a scaling factor of 100 to avoid exploding gradients
                l_input_regions = ExpressionLayer(
                    l_input_reg, lambda X: X[:, :, ::CFG['CONV_REDUCED'], ::CFG['CONV_REDUCED']], output_shape='auto')
            else:
                l_input_regions = l_input_reg
            if CFG['CONV_NORMALIZED'] == 1:
                l_input_regions = ExpressionLayer(
                    l_input_regions, lambda X: X / (T.sum(X, axis=1, keepdims=True) + 1e-8), output_shape='auto')
            elif CFG['CONV_NORMALIZED'] == 2:
                l_input_regions = ExpressionLayer(
                    l_input_regions, lambda X: X / T.sqrt(T.sum(X**2, axis=1, keepdims=True) + 1e-8), output_shape='auto')
            else:
                l_input_regions = ExpressionLayer(
                    l_input_regions, lambda X: X * 0.01, output_shape='auto')
            if CFG['TENSOR_ADD_CONV']:
                l_input_regions = ConvLayer(l_input_regions, num_filters=CFG['REGION_SIZE'], filter_size=(
                    3, 3), pad='same', name='l_add_con')
            l_input_regions = ReshapeLayer(
                l_input_regions, (CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS']))

        if CFG['TENSOR_TIED']:
            l_region_feedback = InputLayer((CFG['BATCH_SIZE'], CFG['NUM_REGIONS']), name='l_region_feedback')
            l_region_feedback2 = ReshapeLayer(
                l_region_feedback, ([0], 1, [1]), name='l_region_feedback2')
        else:
            l_shp2 = ReshapeLayer(
                l_prev_gru, (CFG['BATCH_SIZE'], 1, CFG['EMBEDDING_SIZE']))
            l_shp2 = DropoutLayer(
                l_shp2, p=CFG['RNN_DROPOUT'], name='l_shp2')
            l_tensor2 = TensorProdFactLayer((l_shp2, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
                vocab), nonlinearity=lasagne.nonlinearities.softmax, name='l_tensor2')
            l_region_feedback = ExpressionLayer(l_tensor2, lambda X: T.sum(
                X, 3), output_shape='auto', name='l_region')  # sum over
            l_region_feedback2 = ReshapeLayer(
                l_region_feedback, (CFG['BATCH_SIZE'], 1, CFG['NUM_REGIONS']))

        l_weighted_region = WeightedSumLayer(
            [l_input_regions, l_region_feedback2], name='l_weighted_region')
        # define a cell
        l_cell_input = InputLayer((CFG['BATCH_SIZE'], CFG['EMBEDDING_SIZE']), name='l_cell_input')
        if CFG['FEEDBACK'] == 0:  # none
            l_cell_concat = l_cell_input
        elif CFG['FEEDBACK'] == 1:  # none
            l_region2 = ReshapeLayer(
                l_region_feedback2, ([0], [2]))
            l_cell_concat = lasagne.layers.ConcatLayer(
                [l_cell_input, l_region2], axis=1, name='l_cell_concat')
        elif CFG['FEEDBACK'] == 2:
            l_cell_concat = lasagne.layers.ConcatLayer(
                [l_cell_input, l_weighted_region], axis=1, name='l_cell_concat')
        elif CFG['FEEDBACK'] == 3:
            l_region2 = ReshapeLayer(
                l_region_feedback2, ([0], [2]))
            l_cell_concat = lasagne.layers.ConcatLayer(
                [l_cell_input, l_weighted_region, l_region2], axis=1, name='l_cell_concat')
        elif CFG['FEEDBACK'] == 4:
            # See RNNTraining.py for comments on this.
            from TProd3 import WeightedImageLayer
            l_weighted_image = WeightedImageLayer(
                [l_input_regions, l_region_feedback2], name='l_weighted_image')
            if CFG['IMGFEEDBACK_MECHANISM'] == 'highres':
                l_weighted_image_reshaped = ReshapeLayer(
                    l_weighted_image, ([0], [1], 14, 14), name='l_weighted_image_reshaped')
                l_weighted_image_conv_reduced = lasagne.layers.MaxPool2DLayer(
                    l_weighted_image_reshaped, (2, 2), name='l_weighted_image_conv_reduced')
                l_feedback_co1 = lasagne.layers.Conv2DLayer(
                    incoming=l_weighted_image_conv_reduced, num_filters=512, filter_size=(3, 3), pad='same', name='l_feedback_co1')
            else:
                l_weighted_image_reshaped = ReshapeLayer(
                    l_weighted_image, ([0], [1], 7, 7), name='l_weighted_image_reshaped')
                l_feedback_co1 = lasagne.layers.Conv2DLayer(
                    incoming=l_weighted_image_reshaped, num_filters=512, filter_size=(3, 3), pad='same', name='l_feedback_co1')

            l_feedback_po1 = lasagne.layers.MaxPool2DLayer(
                l_feedback_co1, (2, 2), name='l_feedback_po1')
            l_feedback_co2 = lasagne.layers.Conv2DLayer(
                incoming=l_feedback_po1, num_filters=512, filter_size=(3, 3), pad='same', name='l_feedback_co2')
            l_feedback_po2 = lasagne.layers.MaxPool2DLayer(
                l_feedback_co2, (2, 2), name='l_feedback_po2')
            l_feedback_po2_reshaped = ReshapeLayer(
                l_feedback_po2, ([0], [1]), name='l_feedback_po2_reshaped')
            l_cell_concat = lasagne.layers.ConcatLayer(
                [l_cell_input, l_feedback_po2_reshaped], axis=1, name='l_cell_concat')

        from agentnet.memory import GRUMemoryLayer
        l_gru = GRUMemoryLayer(CFG['EMBEDDING_SIZE'],
                               l_cell_concat, l_prev_gru, name='l_gru')
        l_dropout_output = DropoutLayer(
            l_gru, p=0.5, name='l_dropout_output')
        l_shp1 = ReshapeLayer(
            l_dropout_output, ([0], 1, [1]), name='l_shp1')

        if CFG.has_key('DISSECT') and CFG['DISSECT'] != 'No':
            import pdb
            pdb.set_trace()  # XXX BREAKPOINT
            if CFG['DISSECT'] == 'wr':
                l_decoder = TensorProdFactLayer((l_shp1, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'],
                                                dim_w=len(vocab), nonlinearity=softmax, name='l_tensor', W_hr='skip', b_hr='skip')
            elif CFG['DISSECT'] == 'rs':
                import pdb
                pdb.set_trace()  # XXX BREAKPOINT
                l_decoder = TensorProdFactLayer((l_shp1, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'],
                                                dim_w=len(vocab), nonlinearity=softmax, name='l_tensor', W_rw='skip', b_rw='skip')
        else:
            if CFG.has_key('DENSITY_TEMPERING') and CFG['DENSITY_TEMPERING']:
                print("TEMPERING")
                l_gamma = DenseLayer(
                    l_shp1, num_units=1, name='l_gamma')
                l_gamma_shp = ReshapeLayer(
                    l_gamma, ([0], [1], 1, 1))
                from TProd3 import TensorTemperatureLayer
                l_decoder = TensorTemperatureLayer((l_shp1, l_input_regions, l_gamma_shp), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
                    vocab), nonlinearity=lasagne.nonlinearities.softmax, name='l_tensor')

            else:
                l_decoder = TensorProdFactLayer((l_shp1, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'],
                                                dim_w=len(vocab), nonlinearity=softmax, name='l_tensor')

        if CFG['TENSOR_COND_WORD']:
            from RNNTraining import get_Regions_cond_words
            l_region = ExpressionLayer(
                l_decoder, get_Regions_cond_words, output_shape='auto', name='l_region')
        else:
            l_region = ExpressionLayer(l_decoder, lambda X: X.sum(
                3), output_shape='auto', name='l_region')  # sum over
        l_region = ReshapeLayer(
            l_region, ([0], [2]), name='l_region')
        l_out = ExpressionLayer(l_decoder, lambda X: X.sum(
            2), output_shape='auto', name='l_out')  # sum over regions
        #}}}

    elif CFG['MODE'] == 'tensor-reducedw':
        # {{{2
        from TProd3 import TensorProdFactLayer
        # input:  [h(batch,dimh),r(num_batch,r_dim,num_r)]
        # output: [ h[0],r[2], dim_w ]
        l_input_regions = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regins')

        if CFG.has_key('TENSOR_RECTIFY') and CFG['TENSOR_RECTIFY']:
            l_tensor = TensorProdFactLayer((l_dropout_output, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG[
                                           'REGION_SIZE'], dim_w=CFG['EMBEDDING_WORDS'], nonlinearity=lasagne.nonlinearities.rectify, name='l_tensor')
        else:
            l_tensor = TensorProdFactLayer((l_dropout_output, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG[
                                           'REGION_SIZE'], dim_w=CFG['EMBEDDING_WORDS'], nonlinearity=lasagne.nonlinearities.identity, name='l_tensor')
        # softmax does not accept non-flat layers, then flatten->softmax->reshape
        l_flatten = ReshapeLayer(
            l_decoder, (CFG['BATCH_SIZE'] * 1 * CFG['NUM_REGIONS'], CFG['EMBEDDING_WORDS']), name='l_flatten')
        l_words = DenseLayer(l_flatten, num_units=len(vocab),
                                            nonlinearity=lasagne.nonlinearities.identity, name='l_words')

        l_reshape = ReshapeLayer(
            l_words, (CFG['BATCH_SIZE'] * 1, CFG['NUM_REGIONS'] * len(vocab)), name='l_reshape')

        l_softmax = lasagne.layers.NonlinearityLayer(
            l_reshape, nonlinearity=lasagne.nonlinearities.softmax, name='l_softmax')

        l_reshape1 = ReshapeLayer(
            l_softmax, (CFG['BATCH_SIZE'], 1, CFG['NUM_REGIONS'], len(vocab)), name='l_reshape1')

        l_out = ExpressionLayer(l_reshape1, lambda X: X.sum(
            2), output_shape='auto', name='l_out')  # sum over regions
        # }}}

    elif CFG['MODE'] == 'tensor-removedWrw':
        from TProd3 import TensorProdFact2Layer
        # input:  [h(batch,dimh),r(num_batch,r_dim,num_r)]
        # output: [ h[0],r[2], dim_w ]
        l_input_regions = InputLayer((CFG['BATCH_SIZE'], CFG['REGION_SIZE'], CFG['NUM_REGIONS']), name='l_input_regins')

        l_decoder = TensorProdFact2Layer((l_dropout_output, l_input_regions), dim_h=CFG['EMBEDDING_SIZE'], dim_r=CFG['REGION_SIZE'], dim_w=len(
            vocab), nonlinearity=lasagne.nonlinearities.softmax, name='l_decoder')

        l_out = ExpressionLayer(l_decoder, lambda X: X.sum( 2), output_shape='auto', name='l_out')  # sum over regions

    net_dictionnary = {'loc1': l_loc1, 'input_loc': l_input_loc, 'sel_region2': l_sel_region2,
                       'loc': l_loc, 'conv': l_conv, 'prev': l_prev_gru, 'input': l_cell_input,
                       'gru': l_gru, 'sent': l_input_sentence, 'img': l_input_img, 'img2': l_input_img2,
                       'reg_feedback2': l_region_feedback, 'reg_feedback': l_region, 'reg': l_input_reg,
                       'out_reg': l_out_reg, 'out': l_out, 'cnn': l_cnn_embedding, 'sent_emb': l_sentence_embedding,
                       'decoder': l_decoder, 'boxes': l_boxes, 'weighted_regions_prev': l_weighted_region_prev,
                       'weighted_regions': l_weighted_region}
    return net_dictionnary
# }}}
