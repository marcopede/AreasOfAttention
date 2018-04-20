# ResNet-50, network from the paper:
# "Deep Residual Learning for Image Recognition"
# http://arxiv.org/pdf/1512.03385v1.pdf
# License: see https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/resnet50.pkl

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import ReshapeLayer
from save_layers import set_all_layers_tags
try:
    from SPP import SpatialPyramidPoolingDNNLayer as SPPolling
except:
    print("-- SPP import failed (probably because you do not have a GPU to run)")


def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)
    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer
    names : list of string
        Names of the layers in block
    num_filters : int
        Number of filters in convolution layer
    filter_size : int
        Size of filters in convolution layer
    stride : int
        Stride of convolution layer
    pad : int
        Padding of convolution layer
    use_bias : bool
        Whether to use bias in conlovution layer
    nonlin : function
        Nonlinearity type of Nonlinearity layer
    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = []
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, pad, stride,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return dict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix='', get_linear_layer=False):
    """Creates two-branch residual block
    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer
    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block
    ratio_size : float
        Scale factor of filter size
    has_left_branch : bool
        if True, then left branch contains simple block
    upscale_factor : float
        Scale factor of filter bank at the output of residual block
    ix : int
        Id of residual block
    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = {}

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1,name='res%s' % ix)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify,name='res%s_relu' % ix)

    if get_linear_layer:
        return net, 'res%s_relu' % ix, 'res%s' % ix
    else:
        return net, 'res%s_relu' % ix


def build_model(input_layer,im_size=224, batch_size=None, get_linear_layer=False):
    net = {}
    if input_layer!=None:
        net['input'] = input_layer
    else:
        net['input'] = InputLayer((batch_size, 3, im_size, im_size),name='input')
    #net['input'] = InputLayer((batch_size, 3, im_size, im_size),name='input')
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 3, 2, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False,name='pool1')
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)

    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)

    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)
        #print(parent_layer_name)

    block_size = list('abc')
    for c in block_size:
        #if get_linear_layer:
        #    if c == 'a':
        #        sub_net, parent_layer_name = build_residual_block(
        #            net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        #    elif c =='e':
        #        sub_net, parent_layer_name, parent_linear_layer = build_residual_block(
        #            net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        #    else:
        #        sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        #    net.update(sub_net)
        #else:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
        #print(parent_layer_name)

    # res4f_relu is the last layer to have 14 by 14 feature maps
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False,name='pool5')
    net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None,name='fc1000')
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax,name='prob')

    set_all_layers_tags(net['prob'],conv_net=True)
    return net

#def build_model_RCNN(num_boxes,im_size,pool_dims=7,dropout_value=0.5):
def build_model_RCNN(num_boxes, im_size,pool_dims=7,dropout_value=0.5):
    net = {}
    net['input'] = InputLayer((None, 3, im_size, im_size))
    #net['input'] = InputLayer((batch_size, 3, im_size, im_size),name='input')
    #net['input'] = InputLayer((batch_size, 3, im_size, im_size),name='input')
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 3, 2, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False,name='pool1')
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)

    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)

    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)

    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:

            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    # res4f_relu is the last layer to have 14 by 14 feature maps
    print(parent_layer_name)
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False,name='pool5')



    net['boxes'] = InputLayer((None, 5))
    net['crop'] = SPPolling([net['res4f_relu'],net['boxes']],pool_dims=pool_dims,sp_scale = 1./float(16))
    #net['crop'] = SPPolling([net['pool5'],net['boxes']],pool_dims=pool_dims,sp_scale = 1./float(16))
    net['reshape'] = ReshapeLayer(net['crop'], ([0], -1))

    net['fc1000'] = DenseLayer(net['reshape'], num_units=1000, nonlinearity=None,name='fc1000')
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax,name='prob')
   # net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None,name='fc1000')
   # net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax,name='prob')

    set_all_layers_tags(net['prob'],conv_net=True)
    return net
