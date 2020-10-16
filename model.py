import logging
import keras.backend as K
from keras.layers import Dense, Dropout, Activation, Embedding, Input, Concatenate, Lambda
from keras.models import Model, Sequential
from my_layers import Conv1DWithMasking, Remove_domain_emb, Self_attention, Attention, WeightedSum, MyMeanPool
import numpy as np
from layers.graph_relation import SpectralGraphConvolution
#from keras_contrib.layers import CRF

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Custom CNN kernel initializer
# Use the initialization from Kim et al. (2014) for CNN kernel
def my_init(shape, dtype=K.floatx()):
    return 0.01 * np.random.standard_normal(size=shape)

def create_model(args, vocab, nb_class, overall_maxlen, doc_maxlen_1, doc_maxlen_2, num_relations):

    # Funtion that initializes word embeddings 
    def init_emb(emb_matrix, vocab, emb_file_gen, emb_file_domain):

        print 'Loading pretrained general word embeddings and domain word embeddings ...'

        counter_gen = 0.
        pretrained_emb = open(emb_file_gen)
        for line in pretrained_emb:
            tokens = line.split()
            if len(tokens) != 301:
                continue
            word = tokens[0]
            vec = tokens[1:]
            try:
                emb_matrix[0][vocab[word]][:300] = vec
                counter_gen += 1
            except KeyError:
                pass

        if args.use_domain_emb:
            counter_domain = 0.
            pretrained_emb = open(emb_file_domain)
            for line in pretrained_emb:
                tokens = line.split()
                if len(tokens) != 101:
                    continue
                word = tokens[0]
                vec = tokens[1:]
                try:
                    emb_matrix[0][vocab[word]][300:] = vec
                    counter_domain += 1
                except KeyError:
                    pass

        pretrained_emb.close()
        logger.info('%i/%i word vectors initialized by general embeddings (hit rate: %.2f%%)' % (counter_gen, len(vocab), 100*counter_gen/len(vocab)))
        
        if args.use_domain_emb:
            logger.info('%i/%i word vectors initialized by domain embeddings (hit rate: %.2f%%)' % (counter_domain, len(vocab), 100*counter_domain/len(vocab)))

        return emb_matrix


    # Build model
    logger.info('Building model ...')
    print 'Building model ...'
    print '\n\n'

    vocab_size = len(vocab)

    ###################################
    # Inputs 
    ###################################
    print 'Input layer'
    # sequence of token indices for aspect-level data
    sentence_input = Input(shape=(overall_maxlen,), dtype='int32', name='sentence_input')
    A_in = [Input(shape=(overall_maxlen, overall_maxlen), name='A_in%s'%i) for i in range(num_relations)]
    # gold opinion label for aspect-level data. 
    op_label_input = Input(shape=(overall_maxlen, 3), dtype=K.floatx(), name='op_label_input')
    # probability of sending gold opinion labels at opinion transmission step
    p_gold_op = Input(shape=(overall_maxlen,), dtype=K.floatx(), name='p_gold_op') 
    mask = K.not_equal(sentence_input, 0)

    if args.use_doc:
        # doc_input_1 denotes the data for sentiment classification
        # doc_input_2 denotes the data for domain classification
        doc_input_1 = Input(shape=(doc_maxlen_1,), dtype='int32', name='doc_input_1')
        doc_input_2 = Input(shape=(doc_maxlen_2,), dtype='int32', name='doc_input_2')

    if args.use_bert:
        if args.bert_type =='base':
            hs = 768
        bert_input = Input(shape=(overall_maxlen + 1, hs), dtype=K.floatx(), name='bert_input') # +1 denote +cls

    #########################################
    # Shared word embedding layer 
    #########################################
    print 'Word embedding layer'
    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')

    # aspect-level inputs
    word_embeddings = word_emb(sentence_input) 
    sentence_output = word_embeddings

    # doc-level inputs
    if args.use_doc:
        doc_output_1 = word_emb(doc_input_1)
        # we only use general embedding for domain classification
        doc_output_2 = word_emb(doc_input_2)
        if args.use_domain_emb:
            # mask out the domain embeddings
            doc_output_2 = Remove_domain_emb()(doc_output_2)


    def slice(x, index):
        return x[:, index, :]
    def slice1(x, index):
        return x[:, index:, :]
    expand_dim = Lambda(lambda x: K.expand_dims(x, axis = 1))
    if args.use_bert:
        #code.interact(local=locals())
        bert_inp = Lambda(slice1, arguments={'index': 1})(bert_input)
        bert_cls = Lambda(slice, arguments={'index': 0})(bert_input)
        sentence_output = Concatenate()([sentence_output, bert_inp])
#       if args.use_bert_cls:
        #code.interact(local=locals())
        #bert_cls = bert_input[:,0,:]
        node_num = sentence_output.shape.as_list()[1]
        bert_cls1 = expand_dim(bert_cls)
        bert_cls = Lambda(lambda x: K.tile(x, [1, node_num, 1]))(bert_cls1)

    if args.use_bert_cls == 0 and args.use_bert:
        sentence_output = Concatenate()([sentence_output, bert_cls])
    ######################################
    # Shared GCN + CNN layers
    ######################################
#    iter_gcn = Sequential()
#    iter_gcn.add(SpectralGraphConvolution(150, activation='relu',name='GCN'))
    expand_dim = Lambda(lambda x: K.expand_dims(x, axis = 1))
    share_gcn_dense = Dense(300, activation='relu', name='share_gcn_dense')
    for i in xrange(args.shared_layers):
        print 'Shared GCN layer %s'%i
        sentence_output = Dropout(args.dropout_prob)(sentence_output)
        if args.use_doc:
            doc_output_1 = Dropout(args.dropout_prob)(doc_output_1)
            doc_output_2 = Dropout(args.dropout_prob)(doc_output_2)

        if i == 0:
            gcn_0 = SpectralGraphConvolution(args.gcn_dim, args.relation_dim, activation='relu',name='GCN_0')
            # conv_2 = SpectralGraphConvolution(args.cnn_dim, activation='relu',name='GCN_0_2')
            #expand_dim = Lambda(lambda x: K.expand_dims(x, axis = 1))

            sentence_output_0 = gcn_0([sentence_output] + A_in)
            H = Dropout(args.dropout_prob)(sentence_output_0)

            Global_graph = MyMeanPool(axis=1, smask=mask)(H)
            node_num = H.shape.as_list()[1]
            Gg = expand_dim(Global_graph)
            GG = Lambda(lambda x: K.tile(x, [1, node_num, 1]))(Gg)
            # code.interact(local=locals())
            # HG = Concatenate()([H, GG])
            # sentence_output = HG
            if args.use_cnn:
                conv_1 = Conv1DWithMasking(filters=args.cnn_dim, kernel_size=3, \
                activation='relu', padding='same', kernel_initializer=my_init, name='CNN_0_1')
                conv_2 = Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                activation='relu', padding='same', kernel_initializer=my_init, name='CNN_0_2')
                sentence_output_1 = conv_1(sentence_output)
                sentence_output_2 = conv_2(sentence_output)
                
                if args.use_meanpool:
                    sentence_output = Concatenate()([sentence_output_1, sentence_output_2, GG, H])
                else:
                    sentence_output = Concatenate()([sentence_output_1, sentence_output_2, H])
            else:
                if args.use_meanpool:
                    sentence_output = Concatenate()([GG, H])
                else:
                    sentence_output = H

            if args.use_bert_cls == 1 and args.use_bert:
                sentence_output = Concatenate()([sentence_output, bert_cls])

            if args.use_doc:

                doc_output_1_1 = conv_1(doc_output_1)
                doc_output_1_2 = conv_2(doc_output_1)
                doc_output_1 = Concatenate()([doc_output_1_1, doc_output_1_2])

                doc_output_2_1 = conv_1(doc_output_2)
                doc_output_2_2 = conv_2(doc_output_2)
                doc_output_2 = Concatenate()([doc_output_2_1, doc_output_2_2])
            if args.shared_layers == 1:
                sentence_output = share_gcn_dense(sentence_output)

        else:
            # conv = Conv1DWithMasking(filters=args.cnn_dim/3, kernel_size=5, \
            #   activation='relu', padding='same', kernel_initializer=my_init, name='CNN_%s'%i)
            gcn = SpectralGraphConvolution(args.gcn_dim, args.relation_dim, activation='relu',name='GCN_%s'%i)
            #expand_dim = Lambda(lambda x: K.expand_dims(x, axis = 1))

            # sentence_output1 = conv(sentence_output)
            sentence_output2 = gcn([sentence_output] + A_in)
            H = Dropout(args.dropout_prob)(sentence_output2)

            Global_graph = MyMeanPool(axis=1, smask=mask)(H)
            node_num = H.shape.as_list()[1]
            Gg = expand_dim(Global_graph)
            GG = Lambda(lambda x: K.tile(x, [1, node_num, 1]))(Gg)
            # code.interact(local=locals())
            if args.use_cnn:
                conv = Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                  activation='relu', padding='same', kernel_initializer=my_init, name='CNN_%s'%i)
                sentence_output1 = conv(sentence_output)
                # sentence_output = Concatenate()([sentence_output_1, GG, H])
                if args.use_meanpool:
                    sentence_output = Concatenate()([sentence_output_1, GG, H])
                else:
                    sentence_output = Concatenate()([sentence_output_1, H])
            else:
                if args.use_meanpool:
                    sentence_output = Concatenate()([GG, H])
                else:
                    sentence_output = H

            if args.use_bert_cls == 2 and args.use_bert:
                sentence_output = Concatenate()([sentence_output, bert_cls])

            sentence_output = share_gcn_dense(sentence_output)

            if args.use_doc:
                doc_output_1 = conv(doc_output_1)
                doc_output_2 = conv(doc_output_2)

        word_embeddings = Concatenate()([word_embeddings, sentence_output])

    init_shared_features = sentence_output


    #######################################
    # Define task-specific layers
    #######################################

    # AE specific layers
    aspect_cnn = Sequential()
    for a in xrange(args.aspect_layers):
        print 'Aspect extraction layer %s'%a
        aspect_cnn.add(Dropout(args.dropout_prob))
        aspect_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                  activation='relu', padding='same', kernel_initializer=my_init, name='aspect_cnn_%s'%a))
    aspect_dense = Dense(nb_class, activation='softmax', name='aspect_dense')
    aspect_dense_ = Dense(nb_class, name='aspect_dense_')


    # AS specific layers
    sentiment_cnn = Sequential()
    for b in xrange(args.senti_layers):
        print 'Sentiment classification layer %s'%b
        sentiment_cnn.add(Dropout(args.dropout_prob))
        sentiment_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                  activation='relu', padding='same', kernel_initializer=my_init, name='sentiment_cnn_%s'%b))

    sentiment_att = Self_attention(args.use_opinion, name='sentiment_att')
    sentiment_dense = Dense(3, activation='softmax', name='sentiment_dense')
    # sentiment_dense_ = Dense(3, name='sentiment_dense_')


    if args.use_doc:
        # DS specific layers
        doc_senti_cnn = Sequential()
        for c in xrange(args.doc_senti_layers):
            print 'Document-level sentiment layers %s'%c
            doc_senti_cnn.add(Dropout(args.dropout_prob))
            doc_senti_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                      activation='relu', padding='same', kernel_initializer=my_init, name='doc_sentiment_cnn_%s'%c))

        doc_senti_att = Attention(name='doc_senti_att')
        doc_senti_dense = Dense(3, name='doc_senti_dense')
        # The reason not to use the default softmax is that it reports errors when input_dims=2 due to 
        # compatibility issues between the tf and keras versions used.
        softmax = Lambda(lambda x: K.tf.nn.softmax(x), name='doc_senti_softmax')

        # DD specific layers
        doc_domain_cnn = Sequential()
        for d in xrange(args.doc_domain_layers):
            print 'Document-level domain layers %s'%d 
            doc_domain_cnn.add(Dropout(args.dropout_prob))
            doc_domain_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                      activation='relu', padding='same', kernel_initializer=my_init, name='doc_domain_cnn_%s'%d))

        doc_domain_att = Attention(name='doc_domain_att')
        doc_domain_dense = Dense(1, activation='sigmoid', name='doc_domain_dense')

    # re-encoding layer
    enc = Dense(300, activation='relu', name='enc')


    ####################################################
    # aspect-level operations involving message passing
    ####################################################

    for i in xrange(args.interactions+1):
        print 'Interaction number ', i
        aspect_output = sentence_output
        sentiment_output = sentence_output
        # note that the aspet-level data will also go through the doc-level models
        doc_senti_output = sentence_output
        doc_domain_output = sentence_output

        ### AE ###
        if args.aspect_layers > 0:
            aspect_output = aspect_cnn(aspect_output)
        # concate word embeddings and task-specific output for prediction
        aspect_output = Concatenate()([word_embeddings, aspect_output])
        if args.use_bert_cls == 3 and args.use_bert:
            aspect_output = Concatenate()([aspect_output, bert_cls])
        aspect_output = Dropout(args.dropout_prob)(aspect_output)
        aspect_probs = aspect_dense(aspect_output)

        use_crf = 0
        if use_crf:
            aspect_prob = aspect_dense_(aspect_output)
            aspect_crf = CRF(nb_class, sparse_target=True) # False
            aspect_crf_output = aspect_crf(aspect_prob)
        ### AS ###
        if args.senti_layers > 0:
            sentiment_output = sentiment_cnn(sentiment_output)

        sentiment_output = sentiment_att([sentiment_output, op_label_input, aspect_probs, p_gold_op])
        sentiment_output = Concatenate()([init_shared_features, sentiment_output])
        if args.use_bert_cls == 3 and args.use_bert:
            sentiment_output = Concatenate()([sentiment_output, bert_cls])
        sentiment_output = Dropout(args.dropout_prob)(sentiment_output)
        sentiment_probs = sentiment_dense(sentiment_output)
        # use_crf = 0
        # if use_crf:
        #     sentiment_prob = sentiment_dense_(sentiment_output)
        #     senti_crf = CRF(3, sparse_target=False) # False
        #     senti_crf_output = senti_crf(sentiment_prob)

        if args.use_doc:
            ### DS ###
            if args.doc_senti_layers > 0:
                doc_senti_output = doc_senti_cnn(doc_senti_output)
            # output attention weights with two activation functions
            senti_att_weights_softmax, senti_att_weights_sigmoid = doc_senti_att(doc_senti_output)
            # reshape the sigmoid attention weights, will be used in message passing
            senti_weights = Lambda(lambda x: K.expand_dims(x, axis=-1))(senti_att_weights_sigmoid)

            doc_senti_output = WeightedSum()([doc_senti_output, senti_att_weights_softmax])
            doc_senti_output = Dropout(args.dropout_prob)(doc_senti_output)
            doc_senti_output = doc_senti_dense(doc_senti_output)
            doc_senti_probs = softmax(doc_senti_output)
            # reshape the doc-level sentiment predictions, will be used in message passing
            doc_senti_probs = Lambda(lambda x: K.expand_dims(x, axis=-2))(doc_senti_probs)
            doc_senti_probs = Lambda(lambda x: K.repeat_elements(x, overall_maxlen, axis=1))(doc_senti_probs)

            ### DD ###
            if args.doc_domain_layers > 0:
                doc_domain_output = doc_domain_cnn(doc_domain_output)
            domain_att_weights_softmax, domain_att_weights_sigmoid = doc_domain_att(doc_domain_output)
            domain_weights = Lambda(lambda x: K.expand_dims(x, axis=-1))(domain_att_weights_sigmoid)

            doc_domain_output = WeightedSum()([doc_domain_output, domain_att_weights_softmax])
            doc_domain_output = Dropout(args.dropout_prob)(doc_domain_output)
            doc_domain_probs = doc_domain_dense(doc_domain_output)
            
            # update sentence_output for the next iteration
            sentence_output = Concatenate()([sentence_output, aspect_probs, sentiment_probs, 
                                                doc_senti_probs, senti_weights, domain_weights])

        else:
            # update sentence_output for the next iteration
            #sentence_output = Concatenate()([sentence_output, aspect_probs, sentiment_probs])
            if args.use_prob:
			    sentence_output = Concatenate()([sentence_output, aspect_probs, sentiment_probs])
            else:
                sentence_output = Concatenate()([sentence_output, aspect_output, sentiment_output])
        if args.use_bert_cls == 4 and args.use_bert:
            sentence_output = Concatenate()([sentence_output, bert_cls])
        sentence_output = enc(sentence_output)
    use_crf = 0
    if use_crf:
        aspect_prob = aspect_dense_(aspect_output)
        aspect_crf = CRF(nb_class, sparse_target=True) # False
        aspect_crf_output = aspect_crf(aspect_prob)
        aspect_model = Model(inputs=[sentence_input] + A_in + [op_label_input] + [p_gold_op], outputs=[aspect_crf_output, sentiment_probs])
    else:
        if args.use_bert:
            aspect_model = Model(inputs=[sentence_input] + A_in + [op_label_input] + [p_gold_op] + [bert_input], outputs=[aspect_probs, sentiment_probs])
        else:
            aspect_model = Model(inputs=[sentence_input] + A_in + [op_label_input] + [p_gold_op], outputs=[aspect_probs, sentiment_probs])

    ####################################################
    # doc-level operations without message passing
    ####################################################

    if args.use_doc:
        if args.doc_senti_layers > 0:
            doc_output_1 = doc_senti_cnn(doc_output_1)
        att_1, _ = doc_senti_att(doc_output_1)
        doc_output_1 = WeightedSum()([doc_output_1, att_1])
        doc_output_1 = Dropout(args.dropout_prob)(doc_output_1)
        doc_output_1 = doc_senti_dense(doc_output_1)
        doc_prob_1 = softmax(doc_output_1)

        if args.doc_domain_layers > 0:
            doc_output_2 = doc_domain_cnn(doc_output_2)
        att_2, _ = doc_domain_att(doc_output_2)
        doc_output_2 = WeightedSum()([doc_output_2, att_2])
        doc_output_2 = Dropout(args.dropout_prob)(doc_output_2)
        doc_prob_2 = doc_domain_dense(doc_output_2)

        doc_model = Model(inputs=[doc_input_1, doc_input_2], outputs=[doc_prob_1, doc_prob_2])
       
    else:
        doc_model = None


    ####################################################
    # initialize word embeddings
    ####################################################

    logger.info('Initializing lookup table')


    # Load pre-trained word vectors.
    # To save the loading time, here we load from the extracted subsets of the original embeddings, 
    # which only contains the embeddings of words in the vocab. 
    if args.use_doc:
        emb_path_gen = '../glove/%s_.txt'%(args.domain)
        emb_path_domain = '../domain_specific_emb/%s_.txt'%(args.domain)
    else:
        emb_path_gen = '../glove/%s.txt'%(args.domain)
        emb_path_domain = '../domain_specific_emb/%s.txt'%(args.domain)




    # Load pre-trained word vectors from the orginal large files
    # If you are loading from ssd, the process would only take 1-2 mins
    # If you are loading from hhd, the process would take a few hours at first try, 
    # and would take 1-2 mins in subsequent repeating runs (due to cache performance). 

    # emb_path_gen = '../glove.840B.300d.txt'
    # if args.domain == 'lt':
    #     emb_path_domain = '../laptop_emb.vec'
    # else:
    #     emb_path_domain = '../restaurant_emb.vec'



    aspect_model.get_layer('word_emb').set_weights(init_emb(aspect_model.get_layer('word_emb').get_weights(), vocab, emb_path_gen, emb_path_domain))

    logger.info('  Done')
        ## Optimizaer algorithm
    #

    from optimizers import get_optimizer

    optimizer = get_optimizer(args)
    if args.use_doc == 1 and args.interactions > 0:
        # fix the document-specific parameters when updating aspect model
        aspect_model.get_layer('doc_senti_att').trainable = False
        aspect_model.get_layer('doc_senti_dense').trainable = False
        aspect_model.get_layer('doc_domain_att').trainable = False

    if use_crf:
        aspect_model.compile(optimizer=optimizer, 
                loss=[aspect_crf.loss_function,  'categorical_crossentropy'],
                loss_weights=[1., 1.])
    else:
        aspect_model.compile(optimizer=optimizer, 
                    loss=['categorical_crossentropy',  'categorical_crossentropy'],
                    loss_weights=[1., 1.])
    #aspect_model.summary()

    if args.use_doc == 1:
        doc_model.get_layer('doc_senti_att').trainable = True
        doc_model.get_layer('doc_senti_dense').trainable = True
        doc_model.get_layer('doc_domain_att').trainable = True
        doc_model.get_layer('doc_domain_dense').trainable = True


    if args.use_doc:
        doc_model.compile(optimizer=optimizer, 
                    loss=['categorical_crossentropy',  'binary_crossentropy'],
                    loss_weights=[1., 1.],
                    metrics=['categorical_accuracy', 'accuracy'])

    return aspect_model, doc_model



