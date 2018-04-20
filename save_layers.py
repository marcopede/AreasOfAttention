# vim:fdm=marker
# save layers with names
import lasagne
import pdb
import numpy as np


def add_names(net_dict):
    # {{{
    """
    given the net in a dictionary it adds the name field to each layer
    that corresponds to the key of the corresponding dictionary
    """
    for lname, l in net_dict.items():
        l.name = lname

    # }}}

def add_names_layers_and_params(net_dict):
    # {{{
    """
    given the net in a dictionary it adds the name field to each layer
    that corresponds to the key of the corresponding dictionary,
    and also prepends '[layer name].' to each parameters of the layer.
    """
    for lname, l in net_dict.items():
        l.name = lname
        params = l.get_params()
        for p in params:
            if not "." in p.name: # Name of layer not in name of parameter.
                #print("adding name: " + lname + "." + p.name)
                p.name = lname + "." + p.name
                #print("what we get :" + p.name)
    # }}}

def get_param_dict_tied(layer,new_style=False,check_names=True): #works properly when using tied weights
    # {{{
    """
    generate an ordered sequence of parameter names
    assuming that each layer has a name
    """
    sorted_layers = lasagne.layers.get_all_layers(layer)
    param_names = []
    param_values = []
    for l in sorted_layers:
        params = l.get_params()
        #print params
        if params != [] and l.name==None and check_names:
            print('Error, no name for a layer with parameters!')
            sys.exit()
        for p in params:
            if new_style:
                if p.name==None and check_names:
                    print('Error, no name for a layer with parameters!')
                    sys.exit()
                param_names.append(p.name)
            else:
                param_names.append(l.name+"."+p.name)
            param_values.append(p.get_value())
    param_dict = {param_names[l]: param_values[l] for l in range(len(param_names))}
    return param_dict
    # }}}


def generate_param_names(layer,new_style=False):
    # {{{
    """
    generate an ordered sequence of parameter names
    assuming that each layer has a name
    """
    sorted_layers = lasagne.layers.get_all_layers(layer)
    param_names = []
    for l in sorted_layers:
        params = l.get_params()
        # print params
        for p in params:
            if new_style:
                param_names.append(p.name)
            else:
                param_names.append(l.name+"_"+p.name)
    return param_names
    # }}}

#def get_param_dict(layer):
#    # {{{
#    """
#    generate a parameter dictionary ready to save assuming that each layer
#    has a name
#    """
#    # each layer should have a name
#    names = generate_param_names(layer)
#    params = lasagne.layers.get_all_param_values(layer)
#    param_dict = {names[l]: params[l] for l in range(len(names))}
#    return param_dict
#    # }}}

def get_names(layer, lnames):
    # {{{
    """
    use a dictionary to set the parameters of a net, assuming that each
    layer has a name"
    """
    params = layer.get_params()
    if params != []:
        for p in params:
            lnames.append(layer.name+p.name)
    if 'input_layers' in layer.__dict__:
        for l in layer.input_layers:
            lnames = get_names(l, lnames)
    elif 'input_layer' in layer.__dict__:
        lnames = get_names(layer.input_layer, lnames)

    return lnames
    # }}}


def check_names(layer, new_style=False):
    """
    Check that all layer names are unique.
    """
    names = generate_param_names(layer, new_style=new_style)
    if len(names) != len(np.unique(names)):
        raise ValueError("Warning: Some parameters have the same name!!!")

def check_init(layer, relax=False):
    # {{{
    """
    Check that all layers connected to the given layer have been initialized
    """
    layers = lasagne.layers.helper.get_all_layers(layer)
    for layer in layers:
        params = layer.get_params()
        for p in params:
            if not hasattr(p, 'initialized'):
                print("Warning: {} haven't been initialized!!! This may be normal if you are ussing --dissect option, check it.".format(p.name))
                if not relax:
                    raw_input()
    # }}}


def set_all_layers_tags(layer,treat_as_input=None,verbose=False,**tags):
    # {{{
    """
    Set a flag to all layers connected to the given layer
    """
    layers = lasagne.layers.helper.get_all_layers(layer,treat_as_input)
    for layer in layers:
        params = layer.params#get_params()
        for p,ptags in params.iteritems():
            for tk,tval in tags.iteritems():
                if tval:
                    ptags.add(tk)
                else:
                    if tk in ptags:
                        ptags.remove(tk)
            if verbose:
                print "Parameter",p.name,"Tags",ptags
    # }}}


def set_param_dict(layer, mydict,prefix='',show_layers=True, relax=False):
    # {{{
    """
    use a dictionary to set the parameters of a net, assuming that each
    layer has a name
    """
    used = set()
    layers = lasagne.layers.helper.get_all_layers(layer)
    for layer in layers:
        params = layer.get_params()
        #print "LAYER:",layer.name
        for p in params:
            pname=p.name
            if prefix!='' and pname.find(prefix)==0:
                if show_layers:
                    print('Full Name %s'%pname)
                pname=pname[len(prefix):]

            #print("LOOKING FOR:" + pname)
            if pname in mydict:
                if show_layers:
                    print "Parameters:",pname
                v = mydict[pname]
                if np.any(np.isnan(v)):
                    raise ValueError('Nan values in parameters')

                if p.get_value().shape != v.shape:
                    if 'l_sentence_embedding' in pname or 'l_tensor.W_rw' in pname or 'l_tensor.W_hw':
                        print("Last layer, as well as W_rw and W_hw mappings, will remain uninitialized!")
                    else:
                        print(pname)
                        import pdb; pdb.set_trace()
                        raise ValueError(
                         "mismatch: parameter has shape %r but value to "
                         "set has shape %r" %
                         (p.get_value().shape, v.shape))
                else:
                    p.set_value(v)
                    p.initialized = True
                    used.add(pname)
                    # layer.set_params(mydict[layer.name+p.name])
            else:
                if not relax:
                    print "I Could not find parameter" + pname + "in the current dictionary, exiting"
                    raise ValueError('A parameter was not initialized')
                else:
                    print "Warning: I Could not find parameters in the current dictionary!"
                    print "Parameters ", pname, "will remain un-initialized!"
    return used
    # }}}


# load the net as a dictionnary
# use get_param_dict to get a dictionnary of all the parameters
# store the parameters with pickle

