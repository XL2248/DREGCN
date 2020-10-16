import argparse
import codecs,os,code
import logging
import numpy as np
from time import time
import utils as U
import read as dataset
from evaluation import get_metric
import pickle as pkl
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
#os.environ["CUDA_VISIBLE_DEVICES"] = '5'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = 1
sess = tf.Session(config=config)
KTF.set_session(sess)

# logging.basicConfig(
#                     filename='out.log',
#                     level=logging.INFO,
#                     format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


##############################################################################################################################
# Parse arguments

parser = argparse.ArgumentParser()

# argument related to datasets and data preprocessing
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='res', help="domain of the corpus {res, lt, res_15}")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=20000, help="Vocab size. '0' means no limit (default=20000)")

# hyper-parameters related to network training
parser.add_argument("--r", "--relation_dim", dest="relation_dim", type=int, metavar='<int>', default=100, help="GCN output dimension. '0'     means no CNN layer (default=100)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=80, help="Number of epochs (default=80)")
parser.add_argument("--validation-ratio", dest="validation_ratio", type=float, metavar='<float>', default=0.2, help="The percentage of training data used for validation")
parser.add_argument("--pre-epochs", dest="pre_epochs", type=int, metavar='<int>', default=5, help="Number of pretrain document-level epochs (default=5)")
parser.add_argument("-mr", dest="mr", type=int, metavar='<int>', default=2, help="#aspect-level epochs : #document-level epochs = mr:1")
parser.add_argument("--bert-type", dest="bert_type", type=str, metavar='<str>', default='base', help="domain of the corpus {res, lt, res_15}")

# hyper-parameters related to network structur
parser.add_argument("--g", "--gcndim", dest="gcn_dim", type=int, metavar='<int>', default=100, help="GCN output dimension. '0'     means no CNN layer (default=300)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=400, help="Embeddings dimension (default=dim_general_emb + dim_domain_emb = 400)")
parser.add_argument("--c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=300, help="CNN output dimension. '0' means no CNN layer (default=300)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability. (default=0.5)")
parser.add_argument("--lr", dest="learning_rate", type=float, metavar='<float>', default=0.0005, help="The learing rate. (default=0.0005)")

parser.add_argument("--use-bert-cls", dest="use_bert_cls", type=int, metavar='<int>', default=0, help="whether to exploit bert")
parser.add_argument("--bert-layers", dest="bert_layers", type=int, metavar='<int>', default=1, help="The number of bert layers in the shared network")
parser.add_argument("--bert-mode", dest="bert_mode", type=str, metavar='<str>', default='mean', help="mean max first")
parser.add_argument("--only-iter", dest="only_iter", type=int, metavar='<int>', default=0, help="whether to exploit knowledge from document-level data")
parser.add_argument("--only-pretrain", dest="only_pretrain", type=int, metavar='<int>', default=0, help="whether to exploit knowledge from document-level data")
parser.add_argument("--use-prob", dest="use_prob", type=int, metavar='<int>', default=0, help="whether to exploit knowledge from document-level data")
parser.add_argument("--use-bert", dest="use_bert", type=int, metavar='<int>', default=0, help="whether to exploit bert")
             
parser.add_argument("--use-meanpool", dest="use_meanpool", type=int, metavar='<int>', default=1, help="whether to exploit knowledge from document-level data")
parser.add_argument("--use-doc", dest="use_doc", type=int, metavar='<int>', default=0, help="whether to exploit knowledge from document-level data")
parser.add_argument("--train-op", dest="train_op", type=int, metavar='<int>', default=1, help="whether to extract opinion terms")
parser.add_argument("--use-opinion", dest="use_opinion", type=int, metavar='<int>', default=1, help="whether to perform opinion transmission")
parser.add_argument("--use-cnn", dest="use_cnn", type=int, metavar='<int>', default=1, help="whether to exploit cnn in shared layers")
#parser.add_argument("--use-prob", dest="use_prob", type=int, metavar='<int>', default=0, help="whether to exploit knowledge from document-level data")

parser.add_argument("--shared-layers", dest="shared_layers", type=int, metavar='<int>', default=2, help="The number of CNN layers in the shared network")
parser.add_argument("--doc-senti-layers", dest="doc_senti_layers", type=int, metavar='<int>', default=0, help="The number of CNN layers for extracting document-level sentiment features")
parser.add_argument("--doc-domain-layers", dest="doc_domain_layers", type=int, metavar='<int>', default=0, help="The number of CNN layers for extracting document domain features")
parser.add_argument("--senti-layers", dest="senti_layers", type=int, metavar='<int>', default=1, help="The number of CNN layers for extracting aspect-level sentiment features")
parser.add_argument("--aspect-layers", dest="aspect_layers", type=int, metavar='<int>', default=1, help="The number of CNN layers for extracting aspect features")
parser.add_argument("--interactions", dest="interactions", type=int, metavar='<int>', default=2, help="The number of interactions")
parser.add_argument("--use-domain-emb", dest="use_domain_emb", type=int, metavar='<int>', default=1, help="whether to use domain-specific embeddings")

# random seed that affects data splits and parameter intializations
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=123, help="Random seed (default=123)")
args = parser.parse_args()

U.print_args(args)
print(vars(args))
from numpy.random import seed
seed(args.seed)
from tensorflow import set_random_seed
set_random_seed(args.seed)

if args.use_domain_emb == 1:
    assert args.emb_dim == 400
else:
    assert args.emb_dim == 300

###############################################################################################################################
## Prepare data
#

from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import copy

def convert_label(label, nb_class, maxlen):
    label_ = np.zeros((len(label), maxlen, nb_class))
    mask = np.zeros((len(label), maxlen))

    for i in range(len(label)):
        for j in range(len(label[i])):
            l = label[i][j]
            label_[i][j][l] = 1
            mask[i][j] = 1
    return label_, mask

def convert_label_sentiment(label, nb_class, maxlen):
    label_ = np.zeros((len(label), maxlen, nb_class))
    for i in range(len(label)):
        for j in range(len(label[i])):
            l = label[i][j]
            # for background word and word with conflict label, set its sentiment label to [0,0,0]
            # such that we don't consider them in the sentiment classification loss
            if l in [1,2,3]:
                label_[i][j][l-1] = 1
    return label_

def shuffle(array_list):
    len_ = len(array_list[0])
    for x in array_list:
        assert len(x) == len_
    p = np.random.permutation(len_)
    # code.interact(local=locals())
    return [x[p] for x in array_list]


def batch_generator(array_list, batch_size):
    batch_count = 0
    n_batch = len(array_list[0]) / batch_size
    # array_list = shuffle(array_list)

    X = array_list[0]
    A = array_list[1]
    split_x = [[] for x in A[0]] # node num blank list
    
    while True:
        if batch_count == n_batch:
            # array_list = shuffle(array_list)
            batch_count = 0
        batch_start = batch_count*batch_size
        batch_end = (batch_count+1)*batch_size
        split_x[0] = X[batch_start:batch_end]
        for i in range(len(A[0]) - 1):
            for j in range(len(A[batch_start:batch_end])):
                split_x[i + 1].append(A[batch_start:batch_end][j][i].toarray())
            split_x[i + 1] = np.array(split_x[i + 1])

        batch_list = [x[batch_start: batch_end] for x in array_list]
        batch_count += 1
        # yield batch_list1
        a = []
        a.append(split_x)
        for i in range(2, len(array_list)):
            a.append(batch_list[i])
        yield a
        split_x = [[] for x in A[0]]

def batch_generator1(array_list, batch_size):
    batch_count = 0
    n_batch = len(array_list[0]) / batch_size
    # array_list = shuffle(array_list)

    while True:
        if batch_count == n_batch:
            array_list = shuffle(array_list)
            batch_count = 0

        batch_list = [x[batch_count*batch_size: (batch_count+1)*batch_size] for x in array_list]
        batch_count += 1
        yield batch_list

def split_dev(array_list, ratio=0.2):
    validation_size = int(len(array_list[0]) * ratio)
    # array_list = shuffle(array_list)
    dev_sets = [x[:validation_size] for x in array_list]
    train_sets = [x[validation_size:] for x in array_list]
    return train_sets, dev_sets


def generator_A(array_list, batch_size):
    batch_count = 0
    n_batch = len(array_list[0]) / batch_size
    # array_list = shuffle(array_list)

    X = array_list[0]
    A = array_list[1]
    split_x = [[] for x in A[0]] # node num blank list
    
    if batch_count == n_batch:
        # array_list = shuffle(array_list)
        batch_count = 0
    batch_start = batch_count*batch_size
    batch_end = (batch_count+1)*batch_size
    split_x[0] = X[batch_start:batch_end]
    for i in range(len(A[0]) - 1):
        for j in range(len(A[batch_start:batch_end])):
            split_x[i + 1].append(A[batch_start:batch_end][j][i].toarray())
        split_x[i + 1] = np.array(split_x[i + 1])

    # batch_list = [x[batch_start: batch_end] for x in array_list]
    batch_count += 1
    batch_list1 = split_x
    # code.interact(local=locals())
    return batch_list1
# load both aspect-level and document-level data
train_x, train_label_target, train_label_opinion, train_label_polarity, \
test_x, test_label_target, test_label_opinion, test_label_polarity,\
    vocab, overall_maxlen, \
doc_res_x, doc_res_y, doc_lt_x, doc_lt_y, doc_res_maxlen, doc_lt_maxlen = dataset.prepare_data(args.domain, args.vocab_size, args.use_doc)


# print aspect-level data statistics
count_target_train, count_polarity_train = dataset.get_statistics(train_label_target, train_label_polarity)
count_opinion_train = dataset.get_statistics(train_label_opinion)
count_target_test, count_polarity_test = dataset.get_statistics(test_label_target, test_label_polarity)
count_opinion_test = dataset.get_statistics(test_label_opinion)
print '\n------------------ Data statistics -----------------'
print 'Training: #sentence %s, #target %s, #opinion term %s, target polarity count: %s' \
            %(str(len(train_x)), str(count_target_train), str(count_opinion_train), str(count_polarity_train))
print 'Test: #sentence %s, #target %s, #opinion term %s, target polarity count: %s\n' \
            %(str(len(test_x)), str(count_target_test), str(count_opinion_test), str(count_polarity_test))



###################################
# prepare aspect-level data
###################################

# combine the information of train_label_target and train_label_opinion into one sequence tags 
# denoted as train_y_aspectif train_op = True
# 1, 2 denotes the begining of and inside of an aspect term;
# 3, 4 denotes the begining of and inside of an opinion term;
# 0 denotes the background tokens.
train_y_aspect = copy.deepcopy(train_label_target)
test_y_aspect = copy.deepcopy(test_label_target)

if args.train_op: 
    nb_class = 5
    for i in range(len(train_label_target)):
        for j in range(len(train_label_target[i])):
            if train_label_target[i][j] == 0 and train_label_opinion[i][j] > 0:
                train_y_aspect[i][j] = train_label_opinion[i][j] + 2 
    for i in range(len(test_label_target)):
        for j in range(len(test_label_target[i])):
            if test_label_target[i][j] ==0 and test_label_opinion[i][j] > 0:
                test_y_aspect[i][j] = test_label_opinion[i][j] + 2
else:
    nb_class = 3


# Pad sequences to the same length for mini-batch processing
train_x = sequence.pad_sequences(train_x, maxlen=overall_maxlen, padding='post', truncating='post')
test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen, padding='post', truncating='post')

# convert the labels to one-hot encodings
train_y_aspect, train_y_mask = convert_label(train_y_aspect, nb_class, overall_maxlen)
test_y_aspect, test_y_mask = convert_label(test_y_aspect, nb_class, overall_maxlen)

train_y_sentiment = convert_label_sentiment(train_label_polarity, 3, overall_maxlen)
test_y_sentiment = convert_label_sentiment(test_label_polarity, 3, overall_maxlen)

# the original opinion labels will only be used for opinion transimission at training phase
train_y_opinion, _ = convert_label(train_label_opinion, 3, overall_maxlen)
test_y_opinion, _ = convert_label(test_label_opinion, 3, overall_maxlen)
if args.domain == 'res_15':
    A, X = pkl.load(open('../A/%s_new.pkl'%args.domain, 'rb'))
    #A, X, Y, Y_opinion, meta = pkl.load(open('../A/%s.pkl'%args.domain, 'rb'))
else:
    A, X, Y, Y_opinion, meta = pkl.load(open('../A/%s.pkl'%args.domain, 'rb'))
path = '../'
# A, X, Y, meta = pkl.load(open('electronic.pkl', 'rb'))
# A, X, Y, meta = pkl.load(open('yelp.pkl', 'rb'))
num_relations = len(A['train'][0]) - 1
A_train = np.array(A['train'])
A_test = np.array(A['test'])
# code.interact(local=locals())
# split the original training data into train and dev sets

###################################
# prepare bert features
###################################
if args.use_bert:
    if args.bert_type == 'large':
        hs = 1024
    else:
        hs = 768
    berts = []
    with codecs.open(path + 'data_preprocessed/%s/train/sentence.txt.%s_bert-%s_%s'%(args.domain, args.bert_type, args.bert_layers, args.bert_mode), 'r') as f:
        for line in f:
            sentence_length = len(line.strip().split('|||'))
            sentence_bert = [[float(value) for value in values.split()] for values in line.strip().split('|||')] + [[0.0] * hs] * (overall_maxlen +1 - sentence_length)
            berts.append(sentence_bert)
            assert len(berts[0][0]) == hs
            #code.interact(local=locals())
    test_berts = []
    with codecs.open(path + 'data_preprocessed/%s/test/sentence.txt.%s_bert-%s_%s'%(args.domain, args.bert_type, args.bert_layers, args.bert_mode), 'r') as f1:
        for line in f1:
            sentence_length = len(line.strip().split('|||'))
            sentence_bert = [[float(value) for value in values.split()] for values in line.strip().split('|||')] + [[0.0] * hs] * (overall_maxlen +1 - sentence_length)
            test_berts.append(sentence_bert)
    
    if args.use_doc:
        yelp_berts = []
        with codecs.open(path + 'data_doc/yelp_large/yelp_text.txt.large_bert-%s_mean'%args.bert_layers, 'r') as f:
            for line in f:
                sentence_length = len(line.strip().split('|||'))
                sentence_bert = [[float(value) for value in values.split()] for values in line.strip().split('|||')] + [[0.0] * 1024] * (83 - sentence_length)
                yelp_berts.append(sentence_bert)
                #assert len(berts[0]) == 1024
                #code.interact(local=locals())
        elec_berts = []
        with codecs.open(path + 'data_doc/electronics_large/electronics_text.txt.large_bert-%s_mean'%args.bert_layers, 'r') as f1:
            for line in f1:
                sentence_length = len(line.strip().split('|||'))
                sentence_bert = [[float(value) for value in values.split()] for values in line.strip().split('|||')] + [[0.0] * 1024] * (83 - sentence_length)
                elec_berts.append(sentence_bert)
        yelp_berts = np.array(yelp_berts)
        elec_berts = np.array(elec_berts)

    berts = np.array(berts)
    test_berts = np.array(test_berts)
    # split the original training data into train and dev sets
    [train_x, A_train, train_berts, train_y_aspect, train_y_sentiment, train_y_opinion, train_y_mask], \
    [dev_x, A_dev, dev_berts, dev_y_aspect, dev_y_sentiment, dev_y_opinion, dev_y_mask] = \
    split_dev([train_x, A_train, berts, train_y_aspect, train_y_sentiment, train_y_opinion, train_y_mask], ratio=args.validation_ratio)
else:
    [train_x, A_train, train_y_aspect, train_y_sentiment, train_y_opinion, train_y_mask], \
    [dev_x, A_dev, dev_y_aspect, dev_y_sentiment, dev_y_opinion, dev_y_mask] = \
    split_dev([train_x, A_train, train_y_aspect, train_y_sentiment, train_y_opinion, train_y_mask], ratio=args.validation_ratio)


###################################
# prepare document-level data
###################################

if args.use_doc:
    # doc_x_1, doc_y_1 used for predicting the sentiment label 
    # doc_x_2, doc_y_2 used for predicting the domain label between res and lt domains 
    doc_x_2 = np.concatenate((
        sequence.pad_sequences(doc_res_x, maxlen=max(doc_res_maxlen, doc_lt_maxlen), padding='post', truncating='post'),
        sequence.pad_sequences(doc_lt_x, maxlen=max(doc_res_maxlen, doc_lt_maxlen), padding='post', truncating='post'),
        ))
    if args.domain in {'res', 'res_15'}:
        doc_x_1 = sequence.pad_sequences(doc_res_x, maxlen=doc_res_maxlen, padding='post', truncating='post')
        doc_y_1 = to_categorical(doc_res_y)
        doc_y_2 = np.concatenate((np.ones((len(doc_res_y), 1)), np.zeros((len(doc_lt_y), 1))))
        doc_maxlen_1 = doc_res_maxlen
    else:
        doc_x_1 = sequence.pad_sequences(doc_lt_x, maxlen=doc_lt_maxlen, padding='post', truncating='post')
        doc_y_1 = to_categorical(doc_lt_y)
        doc_y_2 = np.concatenate((np.zeros((len(doc_res_y), 1)), np.ones((len(doc_lt_y), 1))))
        doc_maxlen_1 = doc_lt_maxlen

    doc_maxlen_2 = max(doc_res_maxlen, doc_lt_maxlen)

    [train_doc_x_1, train_doc_y_1], [dev_doc_x_1, dev_doc_y_1] = split_dev([doc_x_1, doc_y_1], ratio = 0.1)
    [train_doc_x_2, train_doc_y_2], [dev_doc_x_2, dev_doc_y_2] = split_dev([doc_x_2, doc_y_2], ratio = 0.05)

else:
    doc_maxlen_1, doc_maxlen_2 = None, None



##############################################################################################################################
# ## Optimizaer algorithm
# #

# from optimizers import get_optimizer

# optimizer = get_optimizer(args)


###############################################################################################################################
## Building model
#

from model import create_model

aspect_model, doc_model = create_model(args, vocab, nb_class, overall_maxlen, doc_maxlen_1, doc_maxlen_2, num_relations)



###############################################################################################################################
## Training
#

# compute the probability of using gold opinion labels in opinion transmission
# (To alleviate the problem of unreliable predictions of opinion labels sent from AE to AS at opinion transmission step
#  in the early stage of training, we use gold labels as prediction with probability that depends on the number of current epoch)
def get_prob(epoch_count):
    prob = 5/(5+np.exp(epoch_count/5))
    return prob


from tqdm import tqdm

logger.info('--------------------------------------------------------------------------------------------------------------------------')


########################################
# pre-train document-level tasks
########################################


if args.use_doc:
    gen_doc_1 = batch_generator([train_doc_x_1, train_doc_y_1], batch_size=args.batch_size)
    gen_doc_2 = batch_generator([train_doc_x_2, train_doc_y_2], batch_size=args.batch_size)
    batches_per_epoch_doc = len(train_doc_x_2) / args.batch_size

    for ii in xrange(args.pre_epochs):
        t0 = time()
        loss, loss_sentiment, loss_domain, acc_sentiment, acc_domain = 0., 0., 0., 0., 0.

        # for b in tqdm(xrange(batches_per_epoch_doc)):
        for b in xrange(batches_per_epoch_doc):
            batch_x_1, batch_y_1 = gen_doc_1.next()
            batch_x_2, batch_y_2 = gen_doc_2.next()

            loss_, loss_sentiment_, loss_domain_, acc_sentiment_, _, _, acc_domain_ = doc_model.train_on_batch([batch_x_1, batch_x_2], [batch_y_1, batch_y_2])

            loss += loss_ / batches_per_epoch_doc
            loss_sentiment += loss_sentiment_ / batches_per_epoch_doc
            loss_domain +=  loss_domain_ / batches_per_epoch_doc
            acc_sentiment += acc_sentiment_ / batches_per_epoch_doc
            acc_domain += acc_domain_ / batches_per_epoch_doc

        tr_time = time() - t0

        logger.info('Pretrain doc-level model: Epoch %d, train: %is' % (ii, tr_time))
        logger.info('[Train] loss: %.4f, [Sentiment] loss: %.4f, [Domain] loss: %.4f, [Sentiment] acc: %.4f, \
            [Domain] acc: %.4f,'%(loss, loss_sentiment, loss_domain, acc_sentiment, acc_domain))
        print('Pretrain doc-level model: Epoch %d, train: %is' % (ii, tr_time))
        print('[Train] loss: %.4f, [Sentiment] loss: %.4f, [Domain] loss: %.4f, [Sentiment] acc: %.4f, \
            [Domain] acc: %.4f,'%(loss, loss_sentiment, loss_domain, acc_sentiment, acc_domain))
        valid_loss, valid_loss_sentiment, valid_loss_domain, valid_acc_sentiment, _, _, valid_acc_domain = doc_model.evaluate(
            [dev_doc_x_1, dev_doc_x_2], [dev_doc_y_1, dev_doc_y_2], batch_size=50, verbose=1)

        logger.info('[Validation] loss: %.4f, [Sentiment] loss: %.4f, [Domain] loss: %.4f, [Sentiment] acc: %.4f, \
            [Domain] acc: %.4f,'%(valid_loss, valid_loss_sentiment, valid_loss_domain, valid_acc_sentiment, valid_acc_domain))



######################################################
# train aspect model and document model alternatively
######################################################

best_dev_metric = 0
save_model = False
best_test_metric = 0
# best_test_metric = 0
if args.use_bert:
    gen_aspect_bert = batch_generator([train_x, A_train, train_berts, train_y_aspect, train_y_sentiment, train_y_opinion, train_y_mask], batch_size=args.batch_size)
else:
    gen_aspect = batch_generator([train_x, A_train, train_y_aspect, train_y_sentiment, train_y_opinion, train_y_mask], batch_size=args.batch_size)
batches_per_epoch_aspect = len(train_x) / args.batch_size


for ii in xrange(args.epochs):
    t0 = time()
    loss, loss_aspect, loss_sentiment = 0., 0., 0.

    gold_prob = get_prob(ii)
    rnd = np.random.uniform()
    # as epoch increasing, the probability of using gold opinion label descreases.
    if rnd < gold_prob:
        gold_prob = np.ones((args.batch_size, overall_maxlen))
    else:
        gold_prob = np.zeros((args.batch_size, overall_maxlen))

    for b in xrange(batches_per_epoch_aspect):
    # for b in tqdm(xrange(batches_per_epoch_aspect)):
        if args.use_bert:
            batch_x, batch_bert, batch_y_ae, batch_y_as, batch_y_op, batch_mask = gen_aspect_bert.next()
            #code.interact(local=locals())
            loss_, loss_aspect_, loss_sentiment_ = aspect_model.train_on_batch(batch_x+[batch_y_op]+[gold_prob]+[batch_bert], [batch_y_ae, batch_y_as])
        else:
            batch_x, batch_y_ae, batch_y_as, batch_y_op, batch_mask = gen_aspect.next()
            loss_, loss_aspect_, loss_sentiment_ = aspect_model.train_on_batch(batch_x+[batch_y_op]+[gold_prob], [batch_y_ae, batch_y_as])

        loss += loss_ / batches_per_epoch_aspect
        loss_aspect += loss_aspect_ / batches_per_epoch_aspect
        loss_sentiment += loss_sentiment_ / batches_per_epoch_aspect

        if b%args.mr == 0 and args.use_doc:
            batch_x_1, batch_y_1 = gen_doc_1.next()
            batch_x_2, batch_y_2 = gen_doc_2.next()
            doc_model.train_on_batch([batch_x_1, batch_x_2], [batch_y_1, batch_y_2])

    tr_time = time() - t0

    # logger.info('Epoch %d, train: %is' % (ii, tr_time))
    print('Epoch %d, train: %is' % (ii, tr_time))
    print (loss, loss_aspect, loss_sentiment)
    dev_xx = generator_A([dev_x, A_dev], batch_size=len(dev_x))
    if args.use_bert:
        y_pred_aspect, y_pred_sentiment = aspect_model.predict(dev_xx + [dev_y_opinion] + [np.zeros((len(dev_x), overall_maxlen))] + [dev_berts])
    else: 
        y_pred_aspect, y_pred_sentiment = aspect_model.predict(dev_xx+[dev_y_opinion]+[np.zeros((len(dev_x), overall_maxlen))])

    f_aspect, f_opinion, acc_s, f_s, f_absa \
         = get_metric(dev_y_aspect, y_pred_aspect, dev_y_sentiment, y_pred_sentiment, dev_y_mask, args.train_op)
    print(f_aspect, f_opinion, acc_s, f_s, f_absa)
    # logger.info('Validation results -- [Aspect f1]: %.4f, [Opinion f1]: %.4f, [Sentiment acc]: %.4f, [Sentiment f1]: %.4f, [Overall f1]: %.4f' 
    #                     %(f_aspect, f_opinion, acc_s, f_s, f_absa))
   

    if acc_s > best_dev_metric and ii > 20:
        best_dev_metric = acc_s
        save_model = True
    else:
        save_model = False
    test_xx = generator_A([test_x, A_test], batch_size=len(test_x))
    if args.use_bert:
        y_pred_aspect, y_pred_sentiment = aspect_model.predict(test_xx + [test_y_opinion] + [np.zeros((len(test_x), overall_maxlen))] + [test_berts])
    else:
        y_pred_aspect, y_pred_sentiment = aspect_model.predict(test_xx+[test_y_opinion]+[np.zeros((len(test_x), overall_maxlen))])

    f_aspect, f_opinion, acc_s, f_s, f_absa \
         = get_metric(test_y_aspect, y_pred_aspect, test_y_sentiment, y_pred_sentiment, test_y_mask, args.train_op)
    print(f_aspect, f_opinion, acc_s, f_s, f_absa)
    # logger.info('Test results -- [Aspect f1]: %.4f, [Opinion f1]: %.4f, [Sentiment acc]: %.4f, [Sentiment f1]: %.4f, [Overall f1]: %.4f' 
    #                     %(f_aspect, f_opinion, acc_s, f_s, f_absa))

    best_test_metric = acc_s
    if save_model == True:
        print '-------------- Save model --------------'
        print('current best f-i is:', best_test_metric)
        # logger.info('-------------- Save model --------------\n')



