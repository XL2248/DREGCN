#-*-coding:utf-8 -*-
from __future__ import print_function

import _pickle as pkl
import argparse
import numpy as np
import re
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("-d_t", "--dataset_type", type=str, default="lt", # res/res_15/conll_electronics_shuffle/laptop14
                help="Dataset string ('conll03')")
ap.add_argument("-d", "--dataset", type=str, default="BIO",
                help="Dataset string ('conll2003')")
ap.add_argument("-e", "--embeddings", type=str, default="glove/lt_.txt", 
                help="Name of embeddings file in embeddings/, without .gz extension.")
ap.add_argument("-e_d", "--domain_embeddings", type=str, default="domain_specific_emb/lt_.txt", 
                help="Name of embeddings file in embeddings/, without .gz extension.")
ap.add_argument("-w", "--words", type=int, default=-1,
                help="Maximum number of words in the embeddings.")
ap.add_argument("-c", "--case", type=bool, default=False,
                help="If the embeddings are case sensitive.")
		
args = vars(ap.parse_args())

print(args)
use_opinion = 1
if use_opinion:
	from utils_opinion import *
else:
	from utils import *
# Define parameters
DATASET = args['dataset']
EMBEDDINGS = args['embeddings']
EMBEDDINGS_DOMAIN = args['domain_embeddings']
MAX_NUM_WORDS = args['words']
if MAX_NUM_WORDS < 0:
    MAX_NUM_WORDS = None
CASE_SENSITIVE = args['case']

path = "F:/data"
embeddings_str = path + "embedding/" + EMBEDDINGS# + ".gz"
embeddings_str_domain = path + "embedding/" + EMBEDDINGS_DOMAIN# + ".gz"
embedding_path='glove.840B.300d.part.txt' 

meta = pkl.load(open('word2idx_doc4%s.pkl'%(args['dataset_type']), 'rb')) # please refer to line 74-77 in read.py
word2idx = meta['word2idx']
# or this
# word2idx = word2idx_from_embeddings(embedding_path, max_num_words=MAX_NUM_WORDS)

print(word2idx['<pad>'])
graph_preprocessor = GraphPreprocessor(word2idx=word2idx, case_sensitive=CASE_SENSITIVE)#conll_electronics_shuffle_dev
graph_preprocessor.add_split(path + '/data_preprocessed/%s/train/'%(args['dataset_type']) + args['dataset_type'] + '_train.txt', name='train')
graph_preprocessor.add_split(path + '/data_preprocessed/%s/test/'%(args['dataset_type']) + args['dataset_type'] + '_test.txt', name='test')

A = graph_preprocessor.adjacency_matrices()
X = graph_preprocessor.input_data()

word2idx = graph_preprocessor.word2idx
print(word2idx['<pad>'])

idx2word = {v: k for k, v in word2idx.items()}

label2idx = graph_preprocessor.label2idx
idx2label = {v: k for k, v in label2idx.items()}
# use_opinion = 0
if use_opinion:
	Y_opinion = graph_preprocessor.output_data_opinion()
	label2idx_opinion = graph_preprocessor.label2idx_opinion
	idx2label_opinion = {v: k for k, v in label2idx_opinion.items()}

	meta = {'word2idx': word2idx, 'idx2word': idx2word, 'label2idx': label2idx, 'idx2label': idx2label, 'label2idx_opinion': label2idx_opinion, 'idx2label_opinion': idx2label_opinion}
	pkl.dump((A, X), open('F:/data/' + args['dataset_type'] + '_only_A_norm0.5.pkl', 'wb'), protocol=2)
else:
	meta = {'word2idx': word2idx, 'idx2word': idx2word, 'label2idx': label2idx, 'idx2label': idx2label}
	pkl.dump((A), open('F:/data/yelp4' + args['dataset_type'] + '_A.pkl', 'wb'), protocol=2)
