from __future__ import print_function

from keras import activations, initializers
from keras import regularizers
from keras.engine import Layer
from keras.layers import Dropout, Concatenate, Lambda
import code
import tensorflow as tf
import keras.backend as K


class SpectralGraphConvolution(Layer):
    def __init__(self, output_dim, relation_dim, 
                 init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None,
                 b_regularizer=None, bias=True, 
                 self_links=True, consecutive_links=True, 
                 backward_links=True, edge_embedding=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim  # number of features per node
        self.relation_dim = relation_dim

        self.self_links = self_links
        self.consecutive_links = consecutive_links
        self.backward_links = backward_links
        self.edge_embedding = edge_embedding

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.bias = bias
        self.initial_weights = weights

        self.input_dim = None
        self.W = None
        self.b = None
        self.num_nodes = None
        self.num_features = None
        self.num_relations = None
        self.num_adjacency_matrices = None

        super(SpectralGraphConvolution, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (None, features_shape[1], self.output_dim)
        return output_shape

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 3
        self.input_dim = features_shape[1]
        self.num_nodes = features_shape[1]
        self.num_features = features_shape[2]
        self.num_relations = len(input_shapes) - 1

        self.num_adjacency_matrices = self.num_relations

        if self.consecutive_links:
            self.num_adjacency_matrices += 1

        if self.backward_links:
            self.num_adjacency_matrices *= 2

        if self.self_links:
            self.num_adjacency_matrices += 1

        self.W = []
        self.W_edges = []
        #relation_dim = 200
        for i in range(self.num_adjacency_matrices):
            self.W.append(self.add_weight((self.num_features+self.relation_dim, self.output_dim), # shape: (num_features, output_dim)
                                                    initializer=self.init,
                                                    name='{}_W_rel_{}'.format(self.name, i),
                                                    regularizer=self.W_regularizer))

            if self.edge_embedding:
                self.W_edges.append(self.add_weight((self.relation_dim,), # shape: (num_features, output_dim)
                                                        initializer='ones',
                                                        name='{}_W_edge_{}'.format(self.name, i),
                                                        regularizer=self.W_regularizer))

        self.b = self.add_weight((self.input_dim, self.output_dim),
                                        initializer='random_uniform',
                                        name='{}_b'.format(self.name),
                                        regularizer=self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(SpectralGraphConvolution, self).build(input_shapes)
    def compute_mask(self, input_tensor, mask=None):
        # print("Graph:",mask)
        return mask
    def call (self, inputs, mask=None):
        features = inputs[0] # Shape: (None, num_nodes, num_features)
        A = inputs[1:]  # Shapes: (None, num_nodes, num_nodes)

        eye = A[0] * K.zeros(self.num_nodes, dtype='float32') + K.eye(self.num_nodes, dtype='float32')

        # eye = K.eye(self.num_nodes, dtype='float32')

        if self.consecutive_links:
            shifted = tf.manip.roll(eye, shift=1, axis=0)
            A.append(shifted)

        if self.backward_links:
            for i in range(len(A)):
                A.append(K.permute_dimensions(A[i], [0, 2, 1]))

        if self.self_links:
            A.append(eye)

        AHWs = list()
        expand_dim = Lambda(lambda x: K.expand_dims(x, axis = 0))
        for i in range(self.num_adjacency_matrices):
            if self.edge_embedding:
                w = expand_dim(self.W_edges[i])
                w = Lambda(lambda x: K.tile(x, [self.num_nodes, 1]))(w)
                w = expand_dim(w)
                bs = K.shape(features)[0]
                w = Lambda(lambda x: K.tile(x, [bs, 1, 1]))(w)
#                code.interact(local=locals())
                feature = Concatenate()([features, w])
            else:
                feature = features

            HW = K.dot(feature, self.W[i]) # Shape: (None, num_nodes, output_dim)
            AHW = K.batch_dot(A[i], HW) # Shape: (None, num_nodes, num_features)
            AHWs.append(AHW)
        AHWs_stacked = K.stack(AHWs, axis=1) # Shape: (None, num_supports, num_nodes, num_features)
        output = K.max(AHWs_stacked, axis=1) # Shape: (None, num_nodes, output_dim)

        if self.bias:
            output += self.b
        return self.activation(output)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'num_bases': self.num_bases,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
