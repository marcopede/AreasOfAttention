import theano
import lasagne
from lasagne.layers import Layer,MergeLayer
import theano.tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class RegionCondWordLayer(Layer):
    """Condition layer
    Selects the
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
        a layer containg a number represeting b
    -----
    """
    def __init__(self, incoming, **kwargs):
        super(RegionCondWordLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            (batch,seq,regions,words)
        return
            (batch,seq,regions_cond_words)
        """
        X = input
        batch,seq,regions,words = T.shape(input)
        Xw = X.sum(2) #marginalize regions
        X1 = Xw.reshape((batch*seq,words))#consider batch and sequences the same
        m = T.cast(X1.argmax(1),'int32')#for each element find the most liklely word
        #build the 1d indexing because theano does not support multidimensional advanced indexing on GPU
        X2 = X.reshape((batch*seq*regions*words,))
        rng_regions = T.arange(regions)*words
        rng_batch_seq = T.arange(batch*seq)*regions*words
        idx = ((rng_regions[np.newaxis,:]+m[:,np.newaxis]+rng_batch_seq[:,np.newaxis]).reshape((-1,)))
        X3 = (X2[idx].reshape((batch,seq,regions)))
        return X3

    def get_output_shape_for( self, input_shape ):
        return (input_shape[0],input_shape[1],input_shape[2])

class MySoftmaxLayer(Layer):
    """SoftMax Layer with cuDNN options
    Reshape an input layer of shape (B,F) into a shape of (b,F,n)
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
        a layer containg a number represeting b
    -----
    """
    def __init__(self, incoming, algo='accurate', mode='channel' ,**kwargs):
        super(MySoftmaxLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output of the softmax
        """
        op = theano.sandbox.cuda.dnn.GpuDnnSoftmax('bc01', algo, mode)
        return op(input)

    def get_output_shape_for( self, input_shape ):
        return input_shape

class softmax(Layer):
    """SoftMax Layer with cuDNN options
    Reshape an input layer of shape (B,F) into a shape of (b,F,n)
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
        a layer containg a number represeting b
    -----
    """
    def __init__(self, incoming, algo='accurate', mode='channel' ,**kwargs):
        super(MySoftmaxLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output of the softmax
        """
        op = theano.sandbox.cuda.dnn.GpuDnnSoftmax('bc01', algo, mode)
        return op(input)

    def get_output_shape_for( self, input_shape ):
        return input_shape


class tempered_softmax(MergeLayer):
    """SoftMax Layer with the option of changing a "temperature" to make
    the distribution more peaked or more uniform.
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
        a layer containg a number represeting b
    -----
    """
    def __init__(self, incoming, **kwargs):
        super(tempered_softmax, self).__init__(incoming, **kwargs)

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output of the softmax
        """
        input_layer = inputs[0]
        temperature = inputs[1]
        tempered_input = temperature*input_layer
        return res


    def get_output_shape_for( self, input_shapes ):
        input_layer_shape = input_shapes[0]
        return input_layer_shape

class SoftmaxConvLayer(Layer):
    """Multiplex layer
    Reshape an input layer of shape (B,F) into a shape of (b,F,n)
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
        a layer containg a number represeting b
    -----
    """
    def __init__(self, incoming, **kwargs):
        super(SoftmaxConvLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        x = input
        e_x = T.exp(x - x.max(axis=1, keepdims=True))
        out = e_x / e_x.sum(axis=1, keepdims=True)
        return out

    def get_output_shape_for( self, input_shape ):
        return input_shape

class Upsample1DLayer(Layer):
    """Multiplex layer
    Reshape an input layer of shape (B,F) into a shape of (b,F,n)
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
        a layer containg a number represeting b
    -----
    """
    def __init__(self, incoming, stride=2, kernel=3, **kwargs):
        super(Upsample1DLayer, self).__init__(incoming, **kwargs)
        self.stride = stride
        self.offset = kernel/2

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        output_shape = input.shape
        output = T.zeros((output_shape[0],output_shape[1],output_shape[2]*self.stride + self.offset))
        output = T.set_subtensor(output[:,:,self.offset::self.stride],input)
        return output

    def get_output_shape_for( self, input_shape ):
        return (input_shape[0],input_shape[1],input_shape[2]*self.stride+self.offset)


class Multiplex(MergeLayer):
    """Multiplex layer
    Reshape an input layer of shape (B,F) into a shape of (b,F,n)
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
        a layer containg a number represeting b
    -----
    """
    def __init__(self, incoming, stride=2, **kwargs):
        super(Multiplex, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic:
            return input[:,:,::self.stride,::self.stride]
        else:
            output_shape = self.input_shape[2:4]
            output_shape = np.array(output_shape)/self.stride
            idy = T.tile((T.arange(output_shape[0])*self.stride).reshape((output_shape[0],1)),[1,output_shape[1]])
            idx = T.tile((T.arange(output_shape[1])*self.stride).reshape((1,output_shape[1])),[output_shape[0],1])
            sx=idx+self._srng.multinomial(n=self.stride-1,pvals=.5*np.ones(output_shape))
            sy=idy+self._srng.multinomial(n=self.stride-1,pvals=.5*np.ones(output_shape))

            return input[:,:,sy,sx]

class SubsampleLayer(Layer):
    """Subsample layer
    Randomly selects a subset of the data
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    stride : int
        the size of the subasmpling
    -----
    Note: so far works only with even sizes of the conv layers
    """
    def __init__(self, incoming, stride=2, **kwargs):
        super(SubsampleLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.stride = stride

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic:
            return input[:,:,::self.stride,::self.stride]
        else:
            output_shape = self.input_shape[2:4]
            output_shape = np.array(output_shape)/self.stride
            idy = T.tile((T.arange(output_shape[0])*self.stride).reshape((output_shape[0],1)),[1,output_shape[1]])
            idx = T.tile((T.arange(output_shape[1])*self.stride).reshape((1,output_shape[1])),[output_shape[0],1])
            sx=idx+self._srng.multinomial(n=self.stride-1,pvals=.5*np.ones(output_shape))
            sy=idy+self._srng.multinomial(n=self.stride-1,pvals=.5*np.ones(output_shape))

            return input[:,:,sy,sx]

    def get_output_shape_for( self, input_shape ):
        return (input_shape[0],input_shape[1],input_shape[2]/self.stride,input_shape[3]/self.stride)


class MaxRegionsLayer( MergeLayer ):
    """
    WeightedSum layer( self, incoming, **kwargs)

    Produces the region distribution with the selected word

    Parameters
    ----------
    incoming : two :class:`Layer` instances. feat is (batch,feat,region) weights is (batch,1,region)
        The layer feeding into this layer, or the expected input shape.

    dimensions : list
        Defines the tensor dimension

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

    """
    def __init__( self, incoming, **kwargs):
            super( WeightedSumLayer, self ).__init__( incoming, **kwargs )

    def get_output_for( self, inputs ,**kwargs ):
        return T.sum(feat*weight,2)

    def get_output_shape_for( self, input_shapes ):
        input_feat = input_shapes[0]
        input_weight = input_shapes[1]
        assert(input_feat[0]==input_weight[0])
        return ( input_feat[0],input_feat[1] )

class WeightedSum2Layer( MergeLayer ):
    """
    WeightedSum layer( self, incoming, **kwargs)

    Produces the weighted sum of feat by weigths

    Parameters
    ----------
    incoming : two :class:`Layer` instances. feat is (batch,region,h) weights is (batch,region)

    output : (batch, region , h)

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

    """
    def __init__( self, incoming, **kwargs):
            super( WeightedSum2Layer, self ).__init__( incoming, **kwargs )

    def get_output_for( self, inputs ,**kwargs ):
        feat = inputs[0]
        weight = inputs[1]
        return (T.dot(T.sum(feat*weight,1)[:,:,np.newaxis],T.ones((1,weight.shape[1])))).dimshuffle((0,2,1))

    def get_output_shape_for( self, input_shapes ):
        input_feat = input_shapes[0]
        input_weight = input_shapes[1]
        assert(input_feat[0]==input_weight[0])
        return ( input_feat )


class WeightedSumLayer( MergeLayer ):
    """
    WeightedSum layer( self, incoming, **kwargs)

    Produces the weighted sum of feat by weigths

    Parameters
    ----------
    incoming : two :class:`Layer` instances. feat is (batch,feat,region) weights is (batch,1,region)

    output : (batch, feat)

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

    """
    def __init__( self, incoming, **kwargs):
            super( WeightedSumLayer, self ).__init__( incoming, **kwargs )

    def get_output_for( self, inputs ,**kwargs ):
        feat = inputs[0]
        weight = inputs[1]
        return T.sum(feat*weight,2)

    def get_output_shape_for( self, input_shapes ):
        input_feat = input_shapes[0]
        input_weight = input_shapes[1]
        assert(input_feat[0]==input_weight[0])

        assert(input_feat[2]==input_weight[2])
        return ( input_feat[0],input_feat[1] )

class WeightedImageLayer( MergeLayer ):
    """
    WeightedImage layer( self, incoming, **kwargs)

    Weigths each reagion of the feature map by it's corresponding weight

    Parameters
    ----------
    incoming : two :class:`Layer` instances. feat is (batch,feat,region) weights is (batch,1,region)

    output : (batch, feat, region)

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    """
    def __init__( self, incoming, **kwargs):
            super( WeightedImageLayer, self ).__init__( incoming, **kwargs )

    def get_output_for( self, inputs ,**kwargs ):
        feat = inputs[0]
        weight = inputs[1]
        res = feat*weight
        return res

    def get_output_shape_for( self, input_shapes ):
        input_feat = input_shapes[0]
        input_weight = input_shapes[1]
        assert(input_feat[0]==input_weight[0])
        assert(input_feat[2]==input_weight[2])
        return (input_feat[0],input_feat[1], input_feat[2])

class Temper_Tensor( MergeLayer ):
    """
    WeightedImage layer( self, incoming, **kwargs)

    Mutliplies the unnormalized tensor by the temperature.
    Parameters
    ----------
    incoming : two :class:`Layer` instances. feat is (batch,feat,region) weights is (batch,1,region)

    output : (batch, feat, region)

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    """
    def __init__( self, incoming, **kwargs):
            super( Temper_Tensor, self ).__init__( incoming, **kwargs )

    def get_output_for( self, inputs ,**kwargs ):
        feat = inputs[0]
        weight = inputs[1]
        res = feat*weight
        return res
        #return lasagne.layers.nonlinearities.softmax(res)

    def get_output_shape_for( self, input_shapes ):
        input_feat = input_shapes[0]
        input_weight = input_shapes[1]
        assert(input_feat[0]==input_weight[0])
        #assert(input_feat[2]==input_weight[2])
        return (input_feat[0], input_feat[1], input_feat[2], input_feat[3])

class WeightedSumLayerRep( MergeLayer ):
    """
    WeightedSum layer( self, incoming, **kwargs)

    Produces the weighted sum of feat by weigths

    Parameters
    ----------
    incoming : two :class:`Layer` instances. feat is (batch,feat,region) weights is (batch,1,region)

    output : (batch*region, feat)

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

    """
    def __init__( self, incoming, **kwargs):
            super( WeightedSumLayerRep, self ).__init__( incoming, **kwargs )

    def get_output_for( self, inputs ,**kwargs ):
        feat = inputs[0]
        weight = inputs[1]
        wsum = T.sum(feat*weight,2)
        return T.tile(wsum[:,np.newaxis,:],(1,feat.shape[2],1)).reshape((feat.shape[0]*feat.shape[2],feat.shape[1]))

    def get_output_shape_for( self, input_shapes ):
        input_feat = input_shapes[0]
        input_weight = input_shapes[1]
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        assert(input_feat[0]==input_weight[0])


class TensorProdLayer( MergeLayer ):
    """
    TensorProdLayer( self, incoming, **kwargs)

    Produces the tensor product of the 3 input data

    Parameters
    ----------
    incoming : a :class:`Layer` instance and an input set of boxes
        The layer feeding into this layer, or the expected input shape.

    dimensions : list
        Defines the tensor dimension

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

    """
    def __init__( self, incoming, dim_h = 256, dim_r = 512 , dim_w = 1000, W=lasagne.init.Normal(0.01), **kwargs):
            super( TensorProdLayer, self ).__init__( incoming, **kwargs )
            self.dim_h = dim_h
            self.dim_r = dim_r
            self.dim_w = dim_w
            self.W = self.add_param(W, (dim_h, dim_r, dim_w), name='W')

    def get_output_for( self, inputs ,**kwargs ):
        h = inputs[0] # (None,dim_h)
        num_batch = h.shape[0]
        r = inputs[1] # (None,num_reg,dim_r)
        num_r = r.shape[2]
        Whw=T.dot(r.swapaxes(1,2),self.W)
        score = (T.sum(Whw.dimshuffle((1,3,0,2))*h,3)).swapaxes(2,0)
        return score

    def get_output_shape_for( self, input_shapes ):
        input_h = input_shapes[0]
        input_r = input_shapes[1]
        assert(input_h[0]==input_r[0])
        return ( input_h[0],input_h[1], input_r[1] )


class TensorProd2DLayer( MergeLayer ):
    """
    TensorProdLayer( self, incoming, **kwargs)

    Produces the tensor product of the 3 input data

    Parameters
    ----------
    incoming : a :class:`Layer` instance and an input set of boxes
        The layer feeding into this layer, or the expected input shape.

    dimensions : list
        Defines the tensor dimension

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

    """
    def __init__( self, incoming, dim_h = 256, dim_r = 512 , W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
            super( TensorProd2DLayer, self ).__init__( incoming, **kwargs )
            self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
            self.dim_h = dim_h
            self.dim_r = dim_r
            self.W_hr = self.add_param(W, (dim_h, dim_r), name='W_hr')
            if b is None:
                self.b_hr = None
            else:
                self.b_hr = self.add_param(b, (dim_r,), name="b_hr", regularizable=False)#,trainable=False)

    def get_output_for( self, inputs ,**kwargs ):
        """
            input : [h(batch,dimh),r(num_batch,r_dim,num_r)]
            output: (batch,num_r)
        """
        h = inputs[0] # (None,seq,dim_h)
        num_batch = h.shape[0]
        r = inputs[1] # (None,num_reg,dim_r)
        num_r = r.shape[2]
        seq = h.shape[1]
        words = T.dot(h,self.W_hw) + self.b_hw[np.newaxis,:]
        r1 = r.swapaxes(1,2).reshape((-1,self.dim_r))
        regionwords = T.dot(r1,self.W_rw).reshape((num_batch,num_r,-1)) + self.b_rw[np.newaxis,np.newaxis,:]
        r2=r.swapaxes(1,2).swapaxes(0,1)
        h2=T.dot(h,self.W_hr).swapaxes(0,1) + self.b_hr[np.newaxis,np.newaxis,:]
        regions = (h2[np.newaxis,:,:,:]*r2[:,np.newaxis,:,:]).sum(-1).T
        words = words.reshape((-1,self.dim_w))
        regions = regions.reshape((-1,num_r))
        wordregions = (words[:,np.newaxis,:].T+regions.T).T.reshape((num_batch,seq,num_r,self.dim_w))
        score = wordregions+regionwords[:,np.newaxis,:,:] #words * regionwords * regi
        return self.nonlinearity(score.reshape((num_batch*seq,num_r*self.dim_w))).reshape((num_batch,seq,num_r,self.dim_w))

    def get_output_shape_for( self, input_shapes ):
        input_h = input_shapes[0]
        input_r = input_shapes[1]
        assert(input_h[0]==input_r[0])
        return ( input_h[0],input_r[2])


class TensorProdFactLayer( MergeLayer ):
    """
    TensorProdLayer( self, incoming, **kwargs)

    Produces the tensor product of the 3 input data

    Parameters
    ----------
    incoming : a :class:`Layer` instance and an input set of boxes
        The layer feeding into this layer, or the expected input shape.

    dimensions : list
        Defines the tensor dimension

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

    """
    def __init__( self, incoming, dim_h = 256, dim_r = 512 , dim_w = 1000, W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.), W_hr = None, W_rw = None, W_hw = None, b_hr = None, b_rw = None, b_hw = None,  nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
            super( TensorProdFactLayer, self ).__init__( incoming, **kwargs )
            self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
            self.dim_h = dim_h
            self.dim_r = dim_r
            self.dim_w = dim_w
            if W_hr == None:
                self.W_hr = self.add_param(W, (dim_h, dim_r), name='W_hr')
            elif W_hr == 'skip':
                assert(b_hr=='skip'),"Uncoherent skipping of parameters"
                self.W_hr = None
            else:
                self.W_hr = self.add_param(W_hr, (dim_h, dim_r), name='W_hr')

            if W_rw == None:
                self.W_rw = self.add_param(W, (dim_r, dim_w), name='W_rw')
            elif W_rw == 'skip':
                assert(b_rw=='skip'),"Uncoherent skipping of parameters"
                self.W_rw = None
            else:
                self.W_rw = self.add_param(W_rw, (dim_r, dim_w), name='W_rw')

            if W_hw == None:
                self.W_hw = self.add_param(W, (dim_h, dim_w), name='W_hw')
            elif W_hw == 'skip':
                print("You should not be trying to skip W_hw in my opinion. I will not run, see ya!")
                sys.exit()
            else:
                self.W_hw = self.add_param(W_hw, (dim_h, dim_w), name='W_hw')

            if b is None:
                self.b_hr = None
                self.b_rw = None
                self.b_hw = None
            else:

                if b_hr == None:
                    self.b_hr = self.add_param(b, (dim_r,), name="b_hr", regularizable=False)#,trainable=False)
                elif b_hr == 'skip':
                    self.b_hr = None
                else:
                    self.b_hr = self.add_param(b_hr, (dim_r,), name="b_hr", regularizable=False)#,trainable=False)

                if b_rw == None:
                    self.b_rw = self.add_param(b, (dim_w,), name="b_rw", regularizable=False)#,trainable=False)
                elif b_rw == 'skip':
                    self.b_rw = None
                else:
                    self.b_rw = self.add_param(b_rw, (dim_w,), name="b_rw", regularizable=False)#,trainable=False)

                if b_hw == None:
                    self.b_hw = self.add_param(b, (dim_w,), name="b_hw", regularizable=False)#,trainable=False)
                elif b_hw == 'skip':
                    print("You should not be trying to skip W_hw in my opinion. I will not run, see ya!")
                    sys.exit()
                    #self.b_hw = None
                else:
                    self.b_hw = self.add_param(b_hw, (dim_w,), name="b_hw", regularizable=False)#,trainable=False)

    def get_output_for( self, inputs ,**kwargs ):
        """
            input : [h(batch,seq,dimh),r(num_batch,r_dim,num_r)]
            output: (batch,seq,num_r,dim_w)
        """
        h = inputs[0] # (None,seq,dim_h)
        num_batch = h.shape[0]
        r = inputs[1] # (None,num_reg,dim_r)
        num_r = r.shape[2]
        seq = h.shape[1]

        words = T.dot(h,self.W_hw) + self.b_hw[np.newaxis,:]
        words = words.reshape((-1,self.dim_w))
        r1 = r.swapaxes(1,2).reshape((-1,self.dim_r))

        skip_regionwords = (self.W_rw==None)
        skip_region_state = (self.W_hr==None)

        if not skip_regionwords:
            regionwords = T.dot(r1,self.W_rw).reshape((num_batch,num_r,-1)) + self.b_rw[np.newaxis,np.newaxis,:]

        r2=r.swapaxes(1,2).swapaxes(0,1)

        if not skip_region_state:
            h2=T.dot(h,self.W_hr).swapaxes(0,1) + self.b_hr[np.newaxis,np.newaxis,:]
            regions = (h2[np.newaxis,:,:,:]*r2[:,np.newaxis,:,:]).sum(-1).T
            regions = regions.reshape((-1,num_r))
            wordregions = (words[:,np.newaxis,:].T+regions.T).T.reshape((num_batch,seq,num_r,self.dim_w))
        else:
            words = words[:,np.newaxis,:].T
            words = theano.tensor.extra_ops.repeat(words,num_r,axis=1)
            wordregions = words.T.reshape((num_batch,seq,num_r,self.dim_w))

        if not skip_regionwords:
            score = wordregions+regionwords[:,np.newaxis,:,:] #words * regionwords * regi
        else:
            score = wordregions
        return self.nonlinearity(score.reshape((num_batch*seq,num_r*self.dim_w))).reshape((num_batch,seq,num_r,self.dim_w))

    def get_output_shape_for( self, input_shapes ):
        input_h = input_shapes[0]
        input_r = input_shapes[1]
        assert(input_h[0]==input_r[0])
        assert(input_h[2]==self.dim_h)
        assert(input_r[1]==self.dim_r)
        return ( input_h[0],input_h[1],input_r[2], self.dim_w )

class TensorTemperatureLayer( MergeLayer ):
    """
    TensorProdLayer( self, incoming, **kwargs)

    Produces the tensor product of the 3 input data

    Parameters
    ----------
    incoming : a :class:`Layer` instance and an input set of boxes
        The layer feeding into this layer, or the expected input shape.

    dimensions : list
        Defines the tensor dimension

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

    """
    def __init__( self, incoming, dim_h = 256, dim_r = 512 , dim_w = 1000, W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.), W_hr = None, W_rw = None, W_hw = None, b_hr = None, b_rw = None, b_hw = None,  nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
            super( TensorTemperatureLayer, self ).__init__( incoming, **kwargs )
            self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
            self.dim_h = dim_h
            self.dim_r = dim_r
            self.dim_w = dim_w
            if W_hr == None:
                self.W_hr = self.add_param(W, (dim_h, dim_r), name='W_hr')
            else:
                self.W_hr = self.add_param(W_hr, (dim_h, dim_r), name='W_hr')
            if W_rw == None:
                self.W_rw = self.add_param(W, (dim_r, dim_w), name='W_rw')
            else:
                self.W_rw = self.add_param(W_rw, (dim_r, dim_w), name='W_rw')
            if W_hw == None:
                self.W_hw = self.add_param(W, (dim_h, dim_w), name='W_hw')
            else:
                self.W_hw = self.add_param(W_hw, (dim_h, dim_w), name='W_hw')
            if b is None:
                self.b_hr = None
                self.b_rw = None
                self.b_hw = None
            else:
                if b_hr == None:
                    self.b_hr = self.add_param(b, (dim_r,), name="b_hr", regularizable=False)#,trainable=False)
                else:
                    self.b_hr = self.add_param(b_hr, (dim_r,), name="b_hr", regularizable=False)#,trainable=False)
                if b_rw == None:
                    self.b_rw = self.add_param(b, (dim_w,), name="b_rw", regularizable=False)#,trainable=False)
                else:
                    self.b_rw = self.add_param(b_rw, (dim_w,), name="b_rw", regularizable=False)#,trainable=False)
                if b_hw == None:
                    self.b_hw = self.add_param(b, (dim_w,), name="b_hw", regularizable=False)#,trainable=False)
                else:
                    self.b_hw = self.add_param(b_hw, (dim_w,), name="b_hw", regularizable=False)#,trainable=False)

    def get_output_for( self, inputs ,**kwargs ):
        """
            input : [h(batch,seq,dimh),r(num_batch,r_dim,num_r)]
            output: (batch,seq,num_r,dim_w)
        """
        h = inputs[0] # (None,seq,dim_h)
        num_batch = h.shape[0]
        r = inputs[1] # (None,num_reg,dim_r)
        temperature = inputs[2]
        num_r = r.shape[2]
        seq = h.shape[1]
        words = T.dot(h,self.W_hw) + self.b_hw[np.newaxis,:]
        r1 = r.swapaxes(1,2).reshape((-1,self.dim_r))
        regionwords = T.dot(r1,self.W_rw).reshape((num_batch,num_r,-1)) + self.b_rw[np.newaxis,np.newaxis,:]
        r2=r.swapaxes(1,2).swapaxes(0,1)
        h2=T.dot(h,self.W_hr).swapaxes(0,1) + self.b_hr[np.newaxis,np.newaxis,:]
        regions = (h2[np.newaxis,:,:,:]*r2[:,np.newaxis,:,:]).sum(-1).T
        words = words.reshape((-1,self.dim_w))
        regions = regions.reshape((-1,num_r))
        wordregions = (words[:,np.newaxis,:].T+regions.T).T.reshape((num_batch,seq,num_r,self.dim_w))
        score = wordregions+regionwords[:,np.newaxis,:,:] #words * regionwords * regi
        score = score*temperature
        return self.nonlinearity(score.reshape((num_batch*seq,num_r*self.dim_w))).reshape((num_batch,seq,num_r,self.dim_w))

    def get_output_shape_for( self, input_shapes ):
        input_h = input_shapes[0]
        input_r = input_shapes[1]
        assert(input_h[0]==input_r[0])
        assert(input_h[2]==self.dim_h)
        assert(input_r[1]==self.dim_r)
        return ( input_h[0],input_h[1],input_r[2], self.dim_w )

class TensorProdFact2Layer( MergeLayer ):
    """
    TensorProdLayer( self, incoming, **kwargs)

    Produces the tensor product of the 3 input data

    Parameters
    ----------
    incoming : a :class:`Layer` instance and an input set of boxes
        The layer feeding into this layer, or the expected input shape.

    dimensions : list
        Defines the tensor dimension

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

    """
    def __init__( self, incoming, dim_h = 256, dim_r = 512 , dim_w = 1000, W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
            super( TensorProdFact2Layer, self ).__init__( incoming, **kwargs )
            self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
            self.dim_h = dim_h
            self.dim_r = dim_r
            self.dim_w = dim_w
            self.W_hr = self.add_param(W, (dim_h, dim_r), name='W_hr')
            self.W_hw = self.add_param(W, (dim_h, dim_w), name='W_hw')
            if b is None:
                self.b_hr = None
                self.b_hw = None
            else:
                self.b_hr = self.add_param(b, (dim_r,), name="b_hr", regularizable=False)#,trainable=False)
                self.b_hw = self.add_param(b, (dim_w,), name="b_hw", regularizable=False)#,trainable=False)

    def get_output_for( self, inputs ,**kwargs ):
        """
            input : [h(batch,seq,dimh),r(num_batch,r_dim,num_r)]
            output: (batch,seq,num_r,dim_w)
        """
        h = inputs[0] # (None,dim_h)
        num_batch = h.shape[0]
        r = inputs[1] # (None,num_reg,dim_r)
        num_r = r.shape[2]
        seq = h.shape[1]
        words = T.dot(h,self.W_hw) + self.b_hw[np.newaxis,:]
        r2=r.swapaxes(1,2).swapaxes(0,1)
        h2=T.dot(h,self.W_hr).swapaxes(0,1) + self.b_hr[np.newaxis,np.newaxis,:]
        regions = (h2[np.newaxis,:,:,:]*r2[:,np.newaxis,:,:]).sum(-1).T
        words = words.reshape((-1,self.dim_w))
        regions = regions.reshape((-1,num_r))
        score = (words[:,np.newaxis,:].T+regions.T).T.reshape((num_batch,seq,num_r,self.dim_w))
        return self.nonlinearity(score.reshape((num_batch*seq,num_r*self.dim_w))).reshape((num_batch,seq,num_r,self.dim_w))

    def get_output_shape_for( self, input_shapes ):
        input_h = input_shapes[0]
        input_r = input_shapes[1]
        assert(input_h[0]==input_r[0])
        return ( input_h[0],input_r[2], self.dim_w )


class TensorProd3DFactLayer(MergeLayer):
    """
    TensorProdLayer( self, incoming, **kwargs)

    Produces the tensor product of the 3 input data.

    Parameters
    ----------
    incoming : a :class:`Layer` instance and an input set of boxes
        The layer feeding into this layer, or the expected input shape.

    dimensions : list
            The tensor has three dimensions: length of the hidden state
            vetor incoming, length of the region vector incoming, and dim_w_reduced.
            dim_w should be the real length of the dictionnary. Once the tensor product
            has been done (with reduced dimension dim_w_reduced),
            the result is mapped to a space of the right dimension (dim_w).

            It is up to the caller to reduce (or not) the dimensions along the
            hidden state and region coordinates.

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

    """

    def __init__( self, incoming,
                 dim_w, dim_w_reduced, dim_h=64, dim_r=128,
                 W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.),
                 W_map=lasagne.init.GlorotUniform(), # Mapping to expand words
                 b_map=lasagne.init.Constant(0.),
                 W_hrw=None, b_hrw=None, W_expand=None, b_expand=None,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 **kwargs):
        super(TensorProd3DFactLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        # set dimensions
        self.dim_h = dim_h
        self.dim_r = dim_r
        self.dim_w = dim_w  # length of the vocabulary.
        self.dim_w_reduced = dim_w_reduced  # Reduced word dimension for the tensor

        # Initialize the 3D tensor (with reduced dimension for the words)
        if W_hrw is None:
            self.W_hrw = self.add_param(W, (dim_h, dim_r, dim_w_reduced), name='W_hrw')
        if W_expand is None:
            self.W_expand = self.add_param(W_map, (dim_w, dim_w_reduced), name='W_expand')

        if b is None: # Init the biais for the tensor.
            self.b_hrw = None
        else:
            if b_hrw is None:
                self.b_hrw = self.add_param(
                    b, (dim_r, dim_w_reduced), name="b_hrw", regularizable=False)  # ,trainable=False)
            else:
                self.b_hrw = self.add_param(
                    b_hrw, (dim_r, dim_w_reduced), name="b_hrw", regularizable=False)  # ,trainable=False)

        if b_map is None:  # Init the biais for the mapping to a bigger dictionnary.
            self.b_expand = None
        else:
            if b_expand is None:
                self.b_expand = self.add_param(
                    b, (dim_w,), name="b_expand", regularizable=False)  # ,trainable=False)
            else:
                self.b_expand = self.add_param(
                    b_expand, (dim_w,), name="b_expand", regularizable=False)  # ,trainable=False)
        # }}}


    def get_output_for(self, inputs, **kwargs):
        # {{{3
        """
            input : [h(batch,seq,dimh),r(num_batch,r_dim,num_r)]
            output: (batch,seq,num_r,dim_w)
        """


        # Retrieve variables and dimensions from inputs
        h = inputs[0]  # (None,seq,dim_h)
        num_batch = h.shape[0]
        r = inputs[1]  # (None,num_reg,dim_r)
        num_r = r.shape[2]
        seq = h.shape[1]


        # Multiply the tensor by the hidden state vector to obtain a matrix
        self.b_hrw = self.b_hrw.reshape((self.dim_r,self.dim_w_reduced,1,1))
        matrix = T.tensordot(self.W_hrw, h, axes=[[0],[2]]) +\
            self.b_hrw #Add a bias

        matrix = matrix.swapaxes(0,2)
        matrix = matrix[:,:,:,0] # Squeeze a unit dimension.

        # Multiply the matrix by the region vector to obtain a vector
        # TODO This is not the implementation that I would like to use
        words = theano.sandbox.cuda.blas.batched_dot(matrix, r)

        # Go back to the original dictionnary dimension
        words_expanded = T.tensordot(words, self.W_expand, axes=[[1],[1]]) +\
            self.b_expand

        # Apply a softmax to get a probability (or an other nonlinearity if
        # desired)
        reshaped_score =  self.nonlinearity(words_expanded.reshape((num_batch, seq, num_r,
                                              self.dim_w)))
        return reshaped_score
        # }}}

    def get_output_shape_for(self, input_shapes):
        # {{{3
        input_h = input_shapes[0]
        input_r = input_shapes[1]
        assert(input_h[0] == input_r[0])
        return (input_h[0], input_h[1], input_r[2], self.dim_w)
        # }}}
    # }}}


if __name__ == "__main__":
    #how to use bacth in 3 2D tensors
    num_batch=10
    h_dim = 256
    r_dim = 512
    w_dim = 1000
    num_r = 196
    seq = 32
    h = np.ones((num_batch,seq,h_dim),dtype=np.float32)
    r = np.ones((num_batch,r_dim,num_r),dtype=np.float32)
    W_hr = np.ones((h_dim,r_dim),dtype=np.float32)
    W_rw = np.ones((r_dim,w_dim),dtype=np.float32)
    W_hw = np.ones((h_dim,w_dim),dtype=np.float32)

    import time
    t0 = time.time()
    words = np.dot(h,W_hw)
    r1 = r.swapaxes(1,2).reshape((-1,r_dim))
    regionwords = np.dot(r1,W_rw).reshape((num_batch,num_r,-1))
    r2=r.swapaxes(1,2).swapaxes(0,1)
    h2= np.dot(h,W_hr).swapaxes(0,1)
    regions = (h2[np.newaxis]*r2[:,np.newaxis]).sum(-1).T
    words = words.reshape((-1,w_dim))
    regions = regions.reshape((-1,num_r))
    wordregions = (words[:,np.newaxis,:].T+regions.T).T.reshape((num_batch,seq,num_r,w_dim))
    score = wordregions+regionwords[:,np.newaxis] #words * regionwords * regions
    print score.shape
    print time.time() - t0

    import theano
    theano.config.optimizer='None'#'fast_compile'
    theano.config.exception_verbosity='high'
    #theano.config.compute_test_value = 'warn'
    import lasagne
    hl = lasagne.layers.InputLayer((h.shape))
    rl = lasagne.layers.InputLayer((r.shape))
    t = TensorProdFact2Layer((hl,rl),dim_h=h_dim,dim_r=r_dim,dim_w=w_dim,W = lasagne.init.Constant(1.),nonlinearity = lasagne.nonlinearities.identity)

    x_h = T.tensor3()
    x_r = T.tensor3()

    out = lasagne.layers.get_output(t,{hl: x_h,rl: x_r })
    f = theano.function([x_h,x_r],out,on_unused_input='warn')#,mode=theano.compile.mode.Mode(optimizer=None))

    s = f(h,r)
    print s.shape

