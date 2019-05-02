import hls_model as hlsm
import templates
import numpy as np

class BatchNormalizationBinaryTanh(hlsm.Layer):
    ''' Merged Batch Normalization and Binary Tanh layer.
        The mean, variance, beta, gamma parameters are folded into the threshold at which the 
        sign of the input flips after the Binary Tanh activation.
    '''

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims, precision='ap_uint<1>')

        original_name = self.attributes.get('original_name')
        variance = self.model.get_weights_data(original_name, 'moving_variance')
        mean = self.model.get_weights_data(original_name, 'moving_mean')
        gamma = self.model.get_weights_data(original_name, 'gamma')
        beta = self.model.get_weights_data(original_name, 'beta')
        epsilon = self.model.get_weights_data(original_name, 'epsilon')
        threshold = mean - beta * variance / gamma
        self.add_weights_variable(name='threshold', data=threshold, precision=inp.precision)

    def function_cpp(self):
        params = self._default_function_params()
        params['threshold'] = self.get_weights('threshold').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()
        #params['threshold_T'] = self.get_weights()[0].precision
        
        return self._config_template.format(**params)

# Add the layer type to the layer map
hlsm.layer_map['BatchNormalizationBinaryTanh'] = BatchNormalizationBinaryTanh

# Add the templates for config and function
batchnorm_binarytanh_config_template = """struct config{index} : nnet::batchnorm_binarytanh_config{{
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
}};\n"""

templates.config_templates['BatchNormalizationBinaryTanh'] = batchnorm_binarytanh_config_template

batchnorm_binarytanh_function_template = 'nnet::normalize_binary_tanh<{input_t}, {config}>({input}, {output}, {threshold});'
templates.function_templates['BatchNormalizationBinaryTanh'] = batchnorm_binarytanh_function_template

def replace_layer(graph, key, newLayer):
    ''' Replace the layer at key with newLayer. The key remains the same. '''
    # Iterate through the items in the OrderedDict
    for _ in range(len(graph)):
        # Pop the layer fromt he graph, then replace it, either with the newLayer
        # if the key is the replacement key, otherwise with the original layer
        k, v = graph.popitem(False)
        graph[k] = newLayer if k == key else v
    return graph

'''def delete_layer(graph, key):
    deleted = False
    for _ in range(len(graph)):
        k, v = graph.popitem(False)
        print(k, v)
        if k == key:
            del graph[key]
            deleted = True
        else:
            graph[k] = v
            if deleted:
                v.index -= 1
    return graph'''

def optimize_bnn(model):
    pass_bnn_set_types(model)
    pass_bnn_merge_batch_normalization_binary_tanh(model)
    return model

def pass_bnn_merge_batch_normalization_binary_tanh(model):
    print("Merging BatchNormalization followed by BinaryTanh Activation")
    graph = model.graph
    layers = list(graph.items())
    lastmerged = False
    lastdeleted = False
    nmerged = 0
    for il, (key, layer) in enumerate(layers[:-1]):
        nextlayer = layers[il+1][1]
        # If this is BatchNorm layer and next is binary_tanh, then merge
        if isinstance(layer, hlsm.BatchNormalization) and isinstance(nextlayer, hlsm.Activation) and nextlayer.get_attr('activation', False) == 'binary_tanh':
            nmerged += 1
            lastmerged = True
            lastdeleted = False
            attrs = {}
            attrs['name'] = layer.get_attr('name')
            attrs['original_name'] = layer.get_attr('name')
            attrs['class_name'] = 'BatchNormalizationBinaryTanh'
            attrs['n_in'] = layer.get_attr('n_in')
            attrs['n_out'] = attrs['n_in']
            attrs['n_filt'] = layer.get_attr('n_filt')
            # Make a new layer with the new attributes
            bnbt_layer = BatchNormalizationBinaryTanh(model, 'bnbt{}'.format(il), attrs, layer.inputs, nextlayer.outputs)
            # Replace the old BatchNormalization layer with this one
            replace_layer(graph, attrs['name'], bnbt_layer)
        # If a merge was just performed, the activation layer should be skipped (deleted)
        elif lastmerged:
            lastmerged = False
            lastdeleted = True
            #delete_layer(graph, key)
            del graph[key]
         # If no merge was performed, but the activation layer was deleted,
         # need to rename the input variable to the next layer
        else:
            lastmerged = False
            if lastdeleted:
                # Need to rename the input variable to the next layer
                lastdeleted = False
                layer.get_input_variable().name = bnbt_layer.get_output_variable().name
                layer.get_input_variable().type = bnbt_layer.get_output_variable().type
    print("  Merged {} pairs".format(nmerged))
    return model 

def pass_bnn_set_types(model):
    ''' Set the types for variables in the Binary Neural Network model
        Layer I/O variables are already set, this method sets the types for the weights.
    '''
    print("Setting data types for BNN")
    graph = model.graph
    layers = list(graph.items())
    dense_layers = [l for l in layers if l[1].get_attr('class_name') == 'BinaryDense']
    dense_layers[0][1].get_output_variable().precision = dense_layers[0][1].get_input_variable().precision
    # Set the precision output by the linear layers
    def is_linear(l):
        return l.get_attr('class_name', False) == 'Activation' and l.get_attr('activation', False) == 'linear'

    for il, (name, layer) in enumerate(layers):
        if is_linear(layer):
            layer.get_output_variable().precision = layer.get_input_variable().precision

    return model

