from lasagne.layers import MergeLayer
import theano.tensor as T

class SpatialPyramidPoolingDNNLayerPython( MergeLayer ):
    """
    lasagne.layers.SpatialPyramidPoolingDNNLayer( self, incoming, 
    pool_dims=[4,2,1], **kwargs)

    Spatial Pyramid Pooling Layer

    Performs spatial pyramid pooling (SPP) over the input.
    It will turn a 2D input of arbitrary size into an output of fixed dimension.
    Hence, the convolutional part of a DNN can be connected to a dense part with a
    fixed number of nodes even if the dimensions of the input image are unknown.

    The pooling is performed over :math:`l` pooling levels.
    Each pooling level :math:`i` will create :math:`M_i` output features. 
    :math:`M_i` is given by :math:`n_i * n_i`,
    with :math:`n_i` as the number of pooling operation per dimension in level :math:`i`,
    and we use a list of the :math:`n_i`'s as a parameter for SPP-Layer.
    The length of this list is the level of the spatial pyramid.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_dims : list of integers
        The list of :math:`n_i`'s that define the output dimension of each
        pooling level :math:`i`. The length of pool_dims is the level of
        the spatial pyramid.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between the convolutional part of a
    DNN and its dense part. Convolutions can be used for 
    arbitrary input dimensions, but the size of their output will
    depend on their input dimensions. Connecting the output of the
    convolutional to the dense part then usually demands us to fix
    the dimensions of the network's InputLayer.
    The SPPLayer, however, allows us to leave the network input
    dimensions arbitrary.

    References
    ----------
    .. [1] He, Kaiming et al (2015):
           Spatial Pyramid Pooling in Deep Convolutional Networks 
           for Visual Recognition.
           http://arxiv.org/pdf/1406.4729.pdf.
    """
    def __init__( self, incoming, pool_dims=6, sp_scale=1/float(16), **kwargs):
            super( SpatialPyramidPoolingDNNLayer, self ).__init__( incoming, **kwargs )
            self.num_features = 0
            for pool_dim in pool_dims:
                self.num_features += pool_dim * pool_dim
            self.pool_dims = pool_dims
            self.sp_scale = sp_scale
            #self.num_boxes = num_boxes

    def get_output_for( self, inputs ,**kwargs ):
        # For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
        input = inputs[0]
        boxes = inputs[1]
        batch = T.shape (input)[0]
        channels = T.shape (input)[1]
        height = T.shape( input )[2]
        width = T.shape( input )[3]
        num_boxes = T.shape(boxes)[0]
        output = T.zeros((batch * num_boxes , channels, self.num_features))

        for idbb,bb in enumerate(range(num_boxes)):
            batch_ind = bb[0]

            pool_list = []
            #for pool_dim in self.pool_dims:
            start_w = T.clip(T.floor(bb[1] * self.sp_scale),0,width)
            start_h = T.clip(T.floor(bb[2] * self.sp_scale),0,heigth)
            end_w = T.clip(T.ceil(bb[3] * self.sp_scale),0,width)
            end_h = T.clip(T.ceil(bb[4] * self.sp_scale),0,height)

            w = T.max(end_w - start_w +1,1)
            h = T.amx(end_h - start_h +1,1)

            start_samples_y,start_sample_x = T.floor(_meshgrid(start_h,end_h,pool_dims+1,start_w,end_w,pool_dims+1))
            end_samples_y,end_sample_x = T.ceil(_meshgrid(start_h,end_h,pool_dims+1,start_w,end_w,pool_dims+1))

            input[batch_ind,:,np.floor(py):np.ceil(samples_y[idy+1]),np.floor(px):np.ceil(samples_x[idx+1])]
            
            #T.max()

            #for idx,px in enumerate(samples_x[:-1]):
            #    for idy,py in enumerate(samples_y[:-1]):

             #       (pool.dnn_pool( input[batch_ind,:,np.floor(py):np.ceil(samples_y[idy+1]),np.floor(px):np.ceil(samples_x[idx+1])],(0,0),(None,None),'max', (0,0) )).flatten(2)

                #sz_w = ( w - 1 ) // pool_dim
                #sz_h = ( h - 1 ) // pool_dim

                #str_h = w // pool_dim
                #str_w = h // pool_dim

                #pool = dnn.dnn_pool( input[bb[0],:,start_h:end_h+1,start_w:end_w+1], (sz_h,sz_w),                 (str_h,str_w), 'max', (0,0) ).flatten(2)
        pool_list.append( pool )
        output[idbb] = T.transpose(T.concatenate( pool_list, axis=1 )) #not efficient but for the moment is ok!
        #if everything is correct this vector should be ordered as in fast RCNN    
        return output

    def get_output_shape_for( self, input_shapes ):
        input_shape = zip(*input_shapes)[0]
        boxes_shape = zip(*input_shapes)[1]
        return ( input_shape[0] * boxes_shape, input_shape[1], self.num_features )

    def _linspace(start, stop, num):
        # Theano linspace. Behaves similar to np.linspace
        start = T.cast(start, theano.config.floatX)
        stop = T.cast(stop, theano.config.floatX)
        num = T.cast(num, theano.config.floatX)
        step = (stop-start)/(num-1)
        return T.arange(num, dtype=theano.config.floatX)*step+start


    def _meshgrid(starth,endh,height,startw,endw,width):
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
                    _linspace(startw, endw, width).dimshuffle('x', 0))
        y_t = T.dot(_linspace(starth, endh, height).dimshuffle(0, 'x'),
                    T.ones((1, width)))

        x_t_flat = x_t.reshape((1, -1))
        y_t_flat = y_t.reshape((1, -1))
        ones = T.ones_like(x_t_flat)
        grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
        return grid

from roi_pooling import ROIPoolingOp

class SpatialPyramidPoolingDNNLayer( MergeLayer ):
    """
    lasagne.layers.SpatialPyramidPoolingDNNLayer( self, incoming, 
    pool_dims=6, **kwargs)

    Spatial Pyramid Pooling Layer

    Performs spatial pyramid pooling (SPP) over the input.
    Given a set of boxes, it will turn a 2D input of arbitrary size into an output of fixed dimension.
    Hence, the convolutional part of a DNN can be connected to a dense part with a
    fixed number of nodes even if the dimensions of the input image are unknown.

    The pooling is performed over :math:`l` pooling levels.
    Each pooling level :math:`i` will create :math:`M_i` output features. 
    :math:`M_i` is given by :math:`n_i * n_i`,
    with :math:`n_i` as the number of pooling operation per dimension in level :math:`i`,
    and we use a list of the :math:`n_i`'s as a parameter for SPP-Layer.
    The length of this list is the level of the spatial pyramid.

    Parameters
    ----------
    incoming : a :class:`Layer` instance and an input set of boxes
        The layer feeding into this layer, or the expected input shape.

    pool_dims : integer
        Defines the output dimension 

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between the convolutional part of a
    DNN and its dense part. Convolutions can be used for 
    arbitrary input dimensions, but the size of their output will
    depend on their input dimensions. Connecting the output of the
    convolutional to the dense part then usually demands us to fix
    the dimensions of the network's InputLayer.
    The SPPLayer, however, allows us to leave the network input
    dimensions arbitrary.

    References
    ----------
    .. [1] Ross Girshick et. all, fast RCNN
    """
    def __init__( self, incoming, pool_dims=6, sp_scale=1/float(16), **kwargs):
            super( SpatialPyramidPoolingDNNLayer, self ).__init__( incoming, **kwargs )
            self.pool_dims = pool_dims
            self.sp_scale = sp_scale

    def get_output_for( self, inputs ,**kwargs ):
        # For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
        input = inputs[0]
        boxes = inputs[1]
        batch = T.shape (input)[0]
        channels = T.shape (input)[1]
        height = T.shape( input )[2]
        width = T.shape( input )[3]
        num_boxes = T.shape(boxes)[0]
        #output = T.zeros((batch * num_boxes , channels, self.num_features))
        op = ROIPoolingOp(pooled_h=self.pool_dims, pooled_w=self.pool_dims, spatial_scale=self.sp_scale)
        output = op(input, boxes)
        return output[0]

    def get_output_shape_for( self, input_shapes ):
        input_shape = input_shapes[0]
        boxes_shape = input_shapes[1]
        #print input_shape,boxes_shape
        return ( boxes_shape[0], input_shape[1], self.pool_dims, self.pool_dims )

class SPPBatchLayer( MergeLayer ):
    """
    lasagne.layers.SpatialPyramidPoolingDNNLayer( self, incoming, 
    pool_dims=6, **kwargs)

    Spatial Pyramid Pooling Layer

    Performs spatial pyramid pooling (SPP) over the input.
    Given a set of boxes, it will turn a 2D input of arbitrary size into an output of fixed dimension.
    Hence, the convolutional part of a DNN can be connected to a dense part with a
    fixed number of nodes even if the dimensions of the input image are unknown.

    The pooling is performed over :math:`l` pooling levels.
    Each pooling level :math:`i` will create :math:`M_i` output features. 
    :math:`M_i` is given by :math:`n_i * n_i`,
    with :math:`n_i` as the number of pooling operation per dimension in level :math:`i`,
    and we use a list of the :math:`n_i`'s as a parameter for SPP-Layer.
    The length of this list is the level of the spatial pyramid.

    Parameters
    ----------
    incoming : a :class:`Layer` instance and an input set of boxes
        The layer feeding into this layer, or the expected input shape.

    pool_dims : integer
        Defines the output dimension 

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between the convolutional part of a
    DNN and its dense part. Convolutions can be used for 
    arbitrary input dimensions, but the size of their output will
    depend on their input dimensions. Connecting the output of the
    convolutional to the dense part then usually demands us to fix
    the dimensions of the network's InputLayer.
    The SPPLayer, however, allows us to leave the network input
    dimensions arbitrary.

    References
    ----------
    .. [1] Ross Girshick et. all, fast RCNN
    """
    def __init__( self, incoming, pool_dims=6, sp_scale=1/float(16), **kwargs):
            super( SPPBatchLayer, self ).__init__( incoming, **kwargs )
            self.pool_dims = pool_dims
            self.sp_scale = sp_scale

    def get_output_for( self, inputs ,**kwargs ):
        #input = (batch,channels,14,14)
        #boxes = (batch,num_boxes,5)
        #out = (batch,channels,num_boxes)
        # For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
        input = inputs[0]
        boxes = inputs[1]
        #assert(input.shape[0]==boxes.shape[0])
        batch = T.shape (input)[0]
        channels = T.shape (input)[1]
        height = T.shape( input )[2]
        width = T.shape( input )[3]
        num_boxes = T.shape(boxes)[1]#/batch
        _boxes = boxes.dimshuffle((2,1,0)).reshape((5,num_boxes*batch)).dimshuffle((1,0))#((boxes.dimshuffle((2,1,0))).reshape((5,num_boxes*batch))).dimshuffle((1,0))#boxes#.T.reshape((5,num_boxes*batch)).T
        #for bt in range(batch):
        #    _boxes[bt*num_boxes:(bt+1)*num_boxes,0]=bt
        #output = T.zeros((batch * num_boxes , channels, self.num_features))
        op = ROIPoolingOp(pooled_h=self.pool_dims, pooled_w=self.pool_dims, spatial_scale=self.sp_scale)
        output = op(input, _boxes) #num_boxes*batch,channels,height,width --> batch,channels*height*width,num_boxes
        #output = output[0].reshape((batch,num_boxes,channels*self.pool_dims*self.pool_dims)).dimshuffle((0,2,1))
        #output = output[0].reshape((batch*num_boxes,channels*self.pool_dims*self.pool_dims))
        #output = output.dimshuffle((1,0)).reshape((channels*self.pool_dims*self.pool_dims,num_boxes,batch)).dimshuffle((2,0,1))
        return output[0]

    def get_output_shape_for( self, input_shapes ):
        input_shape = input_shapes[0]
        boxes_shape = input_shapes[1]
        #print input_shape,boxes_shape
        #if boxes_shape[0]!=None:
        #    output_shape = (boxes_shape[0] * boxes_shape[1], input_shape[1]*self.pool_dims*self.pool_dims)
        #else:
        #    output_shape = (None, input_shape[1]*self.pool_dims*self.pool_dims)
        #return  output_shape
        #return (input_shape[0], input_shape[1]*self.pool_dims*self.pool_dims, boxes_shape[1])
        return ( boxes_shape[0], input_shape[1], self.pool_dims, self.pool_dims )


