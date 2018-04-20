#modified transformer layer in order to use multiple proposals
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import Layer,MergeLayer
from lasagne import nonlinearities
from lasagne.utils import as_tuple, floatX

class TranslateLayer(Layer):
    """Shift the location where to pull a layer
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the affine tranformation layer
    -----
    """
    def __init__(self, incoming, layer_size, kernel_size, stride=1,zoom=1.0,**kwargs):
        super(TranslateLayer, self).__init__(incoming, **kwargs)
        self.lx,self.ly=layer_size
        self.kx,self.ky=kernel_size
        self.stride = stride
        self.zoom = zoom
        lx,ly = layer_size
        self.ty = T.cast(T.dot(T.arange(0,ly,stride).reshape(((ly+stride-1)/stride,1)),T.ones((1,(lx+stride-1)/stride))).reshape((1,((ly+stride-1)/stride)*((lx+stride-1)/stride))),'float32')
        self.tx = T.cast(T.dot(T.ones(((ly+stride-1)/stride,1)),T.arange(0,lx,stride).reshape((1,(lx+stride-1)/stride))).reshape((1,((ly+stride-1)/stride)*((lx+stride-1)/stride))),'float32')       

        
    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output of the softmax
        """
        #assert(input[-1]==6)
        output = input[:]
        smp_y = (self.ly+self.stride-1)/self.stride
        smp_x = (self.lx+self.stride-1)/self.stride
        batch_size = input.shape[0]/(smp_y*smp_x) 
        otx = output[:,2].reshape((batch_size,-1))
        oty = output[:,5].reshape((batch_size,-1))
        ntx = otx + self.tx #- T.cast(self.kx, 'int32')/2
        nty = oty + self.ty #- T.cast(self.ky, 'int32')/2
        output=T.set_subtensor(output[:,2],ntx.flatten())
        output=T.set_subtensor(output[:,5],nty.flatten())
        #set initilization to [[1,0,tx],[0,1,ty]]
        output=T.set_subtensor(output[:,0],output[:,0]+1.0)
        output=T.set_subtensor(output[:,4],output[:,4]+1.0)
        if self.zoom!=1.0:
            output=T.set_subtensor(output[:,0:2],output[:,0:2]*self.zoom)
            output=T.set_subtensor(output[:,3:5],output[:,3:5]*self.zoom)
        return output#.reshape(batch_size,self.ly*self.lx,6)

    def get_output_shape_for( self, input_shape ):
        smp_y = (self.ly+self.stride-1)/self.stride
        smp_x = (self.lx+self.stride-1)/self.stride
        batch_size = input_shape[0]/(smp_y*smp_x)
        return (batch_size*smp_y*smp_x,6)#(batch_size,self.ly*self.lx,6)

class MultiTransformerLayer(MergeLayer):
    """
    Spatial transformer layer
    The layer applies an affine transformation on the input. The affine
    transformation is parameterized with six learned parameters [1]_.
    The output is interpolated with a bilinear transformation.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.
    localization_network : a :class:`Layer` instance
        The network that calculates the parameters of the affine
        transformation. See the example for how to initialize to the identity
        transform.
    downsample_factor : float or iterable of float
        A float or a 2-element tuple specifying the downsample factor for the
        output image (in both spatial dimensions). A value of 1 will keep the
        original size of the input. Values larger than 1 will downsample the
        input. Values below 1 will upsample the input.
    References
    ----------
    .. [1]  Max Jaderberg, Karen Simonyan, Andrew Zisserman,
            Koray Kavukcuoglu (2015):
            Spatial Transformer Networks. NIPS 2015,
            http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf
    Examples
    --------
    Here we set up the layer to initially do the identity transform, similarly
    to [1]_. Note that you will want to use a localization with linear output.
    If the output from the localization networks is [t1, t2, t3, t4, t5, t6]
    then t1 and t5 determines zoom, t2 and t4 determines skewness, and t3 and
    t6 move the center position.
    >>> import numpy as np
    >>> import lasagne
    >>> b = np.zeros((2, 3), dtype='float32')
    >>> b[0, 0] = 1
    >>> b[1, 1] = 1
    >>> b = b.flatten()  # identity transform
    >>> W = lasagne.init.Constant(0.0)
    >>> l_in = lasagne.layers.InputLayer((None, 3, 28, 28))
    >>> l_loc = lasagne.layers.DenseLayer(l_in, num_units=6, W=W, b=b,
    ... nonlinearity=None)
    >>> l_trans = lasagne.layers.TransformerLayer(l_in, l_loc)
    """
    def __init__(self, incoming, localization_network, kernel_size=(3,3),zero_padding=False, repeat_input=1,
                 **kwargs):
        super(MultiTransformerLayer, self).__init__(
            [incoming, localization_network], **kwargs)
        self.kernel_size = kernel_size
        self.zero_padding = zero_padding
        self.repeat_input = repeat_input # with repeat input no need to repeat 5 times the same image -> save memory (to test)

        input_shp, loc_shp = self.input_shapes

        if loc_shp[-1] != 6 or len(loc_shp) != 2:
            raise ValueError("The localization network must have "
                             "output shape: (batch_size, 6)")
        if len(input_shp) != 4:
            raise ValueError("The input network must have a 4-dimensional "
                             "output shape: (batch_size, num_input_channels, "
                             "input_rows, input_columns)")

    def get_output_shape_for(self, input_shapes):
        shape = input_shapes[0]
        batch = input_shapes[1][0]
        return (batch,shape[1],self.kernel_size[0],self.kernel_size[1])

    def get_output_for(self, inputs, **kwargs):
        # see eq. (1) and sec 3.1 in [1]
        input, theta = inputs
        if self.zero_padding:
            new_input = T.zeros((input.shape[0],input.shape[1],input.shape[2]+2,input.shape[3]+2))
            #same as the previous two lines but 1D, because theano converts to GPU code only 1D 
            if 0:
                r0 = T.arange(new_input.shape[2])            
                r1 = T.arange(new_input.shape[2])            
                r2 = T.arange(1,new_input.shape[2]-1)
                r3 = T.arange(1,new_input.shape[3]-1)
                sel2 = (r3.reshape((r3.shape[0],1))+new_input.shape[3]*r2.reshape((1,r2.shape[0]))).flatten()
                sel1 = (sel2.reshape((sel2.shape[0],1))+new_input.shape[3]*new_input.shape[2]*r1.reshape((1,r1.shape[0]))).flatten()
                sel0 = (sel1.reshape((sel1.shape[0],1))+new_input.shape[3]*new_input.shape[2]*new_input.shape[1]*r0.reshape((1,r0.shape[0]))).flatten()
                new_input = new_input.flatten()
                new_input = T.set_subtensor(new_input[sel0],input.flatten())
                new_input = new_input.reshape(((input.shape[0],input.shape[1],input.shape[2]+2,input.shape[3]+2)))
            else:
                new_input = T.set_subtensor(new_input[:,:,1:-1,1:-1],input)
            new_theta = theta[:]
            new_theta = T.set_subtensor(new_theta[:,2],theta[:,2]+1)
            new_theta = T.set_subtensor(new_theta[:,5],theta[:,5]+1)
            input = new_input
            theta = new_theta
        return _transform_affine(theta, input, self.kernel_size, self.repeat_input)


def _transform_affine(theta, input, kernel_size, repeat_input):
    num_batch, num_channels, height, width = input.shape
    theta = T.reshape(theta, (-1, 2, 3))
    num_batch_t = theta.shape[0]

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    out_height = T.cast(kernel_size[0],'int64')#T.cast(height // downsample_factor[0], 'int64')
    out_width = T.cast(kernel_size[1],'int64')#T.cast(width // downsample_factor[1], 'int64')
    grid = _meshgrid(out_height, out_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = T.dot(theta, grid)
    x_s = T_g[:, 0]
    y_s = T_g[:, 1]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
    input_transformed = _interpolate(
        input_dim, x_s_flat, y_s_flat,
        out_height, out_width,repeat_input)

    output = T.reshape(
        input_transformed, (num_batch_t, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output#T.cast(output,'float32')


def _interpolate(im, x, y, out_height, out_width, repeat_input):
    # *_f are floats
    num_batch, height, width, channels = im.shape
    num_batch_t = x.shape[0]
    proposals = num_batch_t/num_batch*repeat_input
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)

    # clip coordinates to [0, width/height -1]
    x = T.clip(x, 0, width_f - 1)
    y = T.clip(y, 0, height_f - 1)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    #x = (x + 1) / 2 * (width_f - 1)
    #y = (y + 1) / 2 * (height_f - 1)

    # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing. for
    # indexing, we need to take care they do not extend past the image.
    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    x0 = T.cast(x0_f, 'int64')
    y0 = T.cast(y0_f, 'int64')
    x1 = T.cast(T.minimum(x1_f, width_f - 1), 'int64')
    y1 = T.cast(T.minimum(y1_f, height_f - 1), 'int64')

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width, channels]. We need
    # to offset all indices to match the flat version
    dim2 = width
    dim1 = width*height
    base = T.repeat(
        T.arange(num_batch, dtype='int64')*dim1, proposals)#out_height*out_width*proposals)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels for all samples
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    # calculate interpolated values
    wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
    wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
    wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
    wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
    output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
    return output


def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start


def _meshgrid(height, width):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # It is implemented in Theano instead to support symbolic grid sizes.
    # Note: If the image size is known at layer construction time, we could
    # compute the meshgrid offline in numpy instead of doing it dynamically
    # in Theano. However, it hardly affected performance when we tried.
    x_t = T.dot(T.ones((height, 1)),
                _linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_t = T.dot(_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                T.ones((1, width)))

    x_t_flat = T.cast(x_t.reshape((1, -1))*(T.cast(width,'int64')/2),'float32')
    y_t_flat = T.cast(y_t.reshape((1, -1))*(T.cast(height,'int64')/2),'float32')
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid

