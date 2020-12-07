from keras.utils import to_categorical

import gzip
import codecs, re, operator
import numpy as np
import scipy.sparse as sp
import code
import spacy
from spacy.tokens import Doc
from stanfordcorenlp import StanfordCoreNLP

nlp = spacy.load('en_core_web_sm')

np.random.seed(123)
def is_number(token):
    num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
    return bool(num_regex.match(token))
    
class GraphPreprocessor():

    def __init__(self, word2idx, case_sensitive=True):
        self.splits = {}
        self.relations = []
        self.max_num_nodes = 0
        self.word2idx = word2idx
        self.label2idx = {}
        self.label2idx_opinion = {}
        self.case_sensitive = case_sensitive

    def add_split(self, filepath, name):
        sentences, sentences_labels, sentences_opinion_labels = read_conll(filepath)
        print("Generating dependency graphs for {} split...".format(name))
        sentences_dependency_triples = self._sentences_dependency_triples(sentences)
        
        self.splits[name] = {'sentences': sentences, 'sentences_dependency_triples': sentences_dependency_triples, 'sentences_labels': sentences_labels, 'sentences_opinion_labels': sentences_opinion_labels}

        self._update_relations()
        self._update_max_num_nodes()
        self._update_label2idx()
        self._update_label2idx_opinion()

    def get_split(self, name):
        if name in self.splits:
            return self.splits[name]
        return None

    def _sentences_dependency_triples(self, sentences):
        sentences_dependency_triples = []
        for i, sentence in enumerate(sentences):
            print("", end='\r')
            print("Generating dependency graph {}/{}...".format(i + 1, len(sentences)), end='')
            sentences_dependency_triples.append(self._dependency_triples(sentence))
        print("Done")
        return sentences_dependency_triples

    def _dependency_triples_stanfordcore(self, sentence):
        nlp = StanfordCoreNLP('H:/stanford-parser-full-2018-10-17/stanford-corenlp-full-2018-10-05/stanford-corenlp-full-2018-10-05')
        results = nlp.dependency_parse(sentence)

        triples = []
        root_id = results[0][2] - 1

        for token in results[1:]:
            triples.append((token[2]-1, token[0], token[1]-1))
            if token[2] == root_id:
                triples.append((root_id, 'ROOT', root_id))
        return triples

    def _dependency_triples_stanford(self, sentence):
        with open("H:/2019/ABSA/E2E-TBSA-master/data/lt_parse.txt", "r", encoding="utf-8") as f1:
            content1 = f1.readlines()
        sents = {}
        for line in content1:
            k,v = line.strip().split("####")
            sents[k] = v

        triples = []

        token = sents[' '.join(sentence[:-1])]
        for trip in token.split("##"):
            i, dep, head = trip.split(",")
            triples.append((i, dep, head))
        return triples

    def _dependency_triples(self, sentence):
        doc = Doc(nlp.vocab, words=sentence)
        result = nlp.parser(doc)
        triples = []
        for token in result:
            triples.append((token.i, token.dep_, token.head.i))
        return triples

    def _sentence_dicts(self, split_name):
        sentence_dicts = []
        for i in range(len(self.splits[split_name]['sentences'])):
            sentence_dict = {
                'sentence': self.splits[split_name]['sentences'][i],
                'sentence_dependency_triples': self.splits[split_name]['sentences_dependency_triples'][i],
                'sentence_labels': self.splits[split_name]['sentences_labels'][i],
                'sentence_opinion_labels': self.splits[split_name]['sentences_opinion_labels'][i]
            }
            sentence_dicts.append(sentence_dict)
        return sentence_dicts

    def _sentence_adjacency_matrices(self, sentence_dict, symmetric_normalization):
        adj_matrices = []
        adj_matrix = sp.lil_matrix((self.max_num_nodes, self.max_num_nodes), dtype='int8')
        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                adj_matrix[i, j] = 0.5 # if need:

        for relation in self.relations:
            for triple in sentence_dict['sentence_dependency_triples']:
                if triple[1] == relation:
                    adj_matrix[triple[0], triple[2]] = 1
                    adj_matrix[triple[2], triple[0]] = 1

            adj_matrix = adj_matrix.tocsr()
            if symmetric_normalization:
                adj_matrix = self._symmetric_normalization(adj_matrix)
        adj_matrices.append(adj_matrix)

        return adj_matrices

    def _update_relations(self):
        all_relations = []
        for split_name in self.splits.keys():
            for sentence_triples in self.splits[split_name]['sentences_dependency_triples']:
                for triple in sentence_triples:
                    all_relations.append(triple[1])
        self.relations = list(set(all_relations))
        # print("num_relations:", len(self.relations), self.relations)

    def _update_max_num_nodes(self):
        all_lengths = []
        for split_name in self.splits.keys():
            for sentence in self.splits[split_name]['sentences']:
                all_lengths.append(len(sentence))
        self.max_num_nodes = max(all_lengths)
        print("max sentence lengths:", self.max_num_nodes)

    def _update_label2idx(self):
        all_labels = []
        for split_name in self.splits.keys():
            for sentence_labels in self.splits[split_name]['sentences_labels']:
                for label in sentence_labels:
                    all_labels.append(label)
        unique_labels = list(set(all_labels))
        self.label2idx = {v: i for i, v in enumerate(unique_labels)}

    def _update_label2idx_opinion(self):
        all_labels = []
        for split_name in self.splits.keys():
            for sentence_labels in self.splits[split_name]['sentences_opinion_labels']:
                for label in sentence_labels:
                    all_labels.append(label)
        unique_labels = list(set(all_labels))
        self.label2idx_opinion = {v: i for i, v in enumerate(unique_labels)}

    def _lookup_sentence(self, sentence):

        tokens = [0 for x in range(self.max_num_nodes)] 
        for i, word in enumerate(sentence):
            if not self.case_sensitive:
                word = word.lower()
            if is_number(word):
                word = '<num>'
            if word in self.word2idx:
                tokens[i] = self.word2idx[word]
            else:
                tokens[i] = self.word2idx['<unk>']

        tokens = sp.csr_matrix(np.array(tokens))
        return tokens

    def _lookup_sentence_labels(self, sentence_labels):

        label_idx = [self.label2idx['B-POS'] for x in range(self.max_num_nodes)] 
        for i, label in enumerate(sentence_labels):
            label_idx[i] = self.label2idx[label]
        label_idx = sp.csr_matrix(np.array(label_idx))

        return label_idx

    def _lookup_sentence_labels_opinion(self, sentence_labels):
        label_idx = [self.label2idx_opinion['O'] for x in range(self.max_num_nodes)] 

        for i, label in enumerate(sentence_labels):
            label_idx[i] = self.label2idx_opinion[label]
        label_idx = sp.csr_matrix(np.array(label_idx))

        return label_idx

    def _symmetric_normalization(self, A):
        d = np.array(A.sum(1)).flatten() + 1
        d_inv = 1. / d
        d_inv[np.isinf(d_inv)] = 0.
        D_inv = sp.diags(d_inv)
        return D_inv.dot(A)              

    def input_data(self):
        input_data = {k: [] for k in self.splits.keys()}
        for split_name in self.splits.keys():
            for sentence in self.splits[split_name]['sentences']:
                tokens = self._lookup_sentence(sentence)
                input_data[split_name].append(tokens)
            input_data[split_name] = np.array(input_data[split_name])
        return input_data
 
    def output_data(self):
        output_data = {k: [] for k in self.splits.keys()}
        for split_name in self.splits.keys():
            for sentence_labels in self.splits[split_name]['sentences_labels']:
                label_idx = self._lookup_sentence_labels(sentence_labels)
                output_data[split_name].append(label_idx)
            output_data[split_name] = np.array(output_data[split_name])
        return output_data

    def output_data_opinion(self):
        output_data = {k: [] for k in self.splits.keys()}
        for split_name in self.splits.keys():
            for sentence_labels in self.splits[split_name]['sentences_opinion_labels']:
                label_idx = self._lookup_sentence_labels_opinion(sentence_labels)
                output_data[split_name].append(label_idx)
            output_data[split_name] = np.array(output_data[split_name])
        return output_data

    def adjacency_matrices(self, symmetric_normalization=True):
        A = {k: [] for k in self.splits.keys()}
        node_ids = []
        for split_name in self.splits.keys():
            print("Generating adjacency matrix for {} split...".format(split_name))
            for i, sentence_dict in enumerate(self._sentence_dicts(split_name)):
                print("", end='\r')
                print("Generating adjacency matrix {}/{}...".format(i + 1, len(self.splits[split_name]['sentences'])), end='')

                adjacency_matrix = self._sentence_adjacency_matrices(sentence_dict, symmetric_normalization=symmetric_normalization)
                A[split_name].append(adjacency_matrix)
            print("Done")
        return A

def read_conll (filepath):
    sentences = []
    sentences_labels = []
    sentences_opinion_labels = []
    with open (filepath, "r") as f:
        sentence = []
        sentence_labels = []
        sentence_opinion_labels = []
        for i, line in enumerate(f):
            line = ''.join(i for i in line if ord(i) < 128) # Only allow ASCII characters
            line_split = line.split()
            if len(line_split) == 0:
                sentences.append(sentence)
                sentence = []
                sentences_labels.append(sentence_labels)
                sentence_labels = []
                sentences_opinion_labels.append(sentence_opinion_labels)
                sentence_opinion_labels = []
            else:
                sentence.append(line_split[0])
                sentence_labels.append(line_split[1])
                sentence_opinion_labels.append(line_split[2])

    if len(sentence) > 0:
        sentences.append(sentence)
        sentences_labels.append(sentence_labels)
        sentences_opinion_labels.append(sentence_opinion_labels)
    return sentences, sentences_labels, sentences_opinion_labels

def word2idx_from_embeddings(embeddings_str, max_num_words=None):
    with codecs.open(embeddings_str,'r', encoding='UTF-8') as f:
        word2idx = {}
        tokens = []
        for line in f:
            if max_num_words is not None and len(tokens) >= max_num_words:
                break
            line_split = line.strip().split()
            tokens.append(line_split[0])
        word2idx = {v: k for k, v in enumerate(tokens)}
        return word2idx

def merge_embeddings(embeddings_str, embeddings_str_domain):
    counter_gen = 0.
    vocab = []
    word2embed = {}
    y = 0
    v = 0
    with open(embeddings_str, 'r', encoding='UTF-8') as fopen:
        for line in fopen:
            w = line.strip().split(sep=' ')
            word2embed[w[0]] = w[1:]
            vocab.append(w[0])
            y += 1
    fopen.close()

    word2embed1 = {}
    vocab1 = []
    with open(embeddings_str_domain, 'r', encoding='UTF-8') as fopen:
        for line in fopen:
            w = line.strip().split()
            word2embed1[w[0]] = w[1:]
            vocab1.append(w[0])
            v += 1
    word_vectors = []
    fopen.close()
    i = 0
    k = 0
    with open("embedding/glove.840B.300d.part.txt", 'w',encoding='UTF-8') as fopen:
        for index, word in enumerate(set(vocab + vocab1)):
            i += 1
            s = np.zeros(400)
            if word in word2embed:
                k += 1
                s[:300] = np.array(word2embed[word], dtype=np.float32)
            if word in word2embed1:
                # k += 1
                s[300:] = np.array(word2embed1[word], dtype=np.float32)
            fopen.write(word+' ')
            for n in s:
                fopen.write(' {:.5f}'.format(n))
            fopen.write('\n')
    fopen.close()
    print(y,v, k,i)
        # return emb_matrix
def matrix_dimension(embeddings_str):
    with open(embeddings_str,'r', encoding='UTF-8') as f:
        line = next(f)
        return len(line.split()) - 1#matrix的维度

def matrix_from_embeddings(embeddings_str, word2idx):
    embedding_matrix = np.zeros((len(word2idx), matrix_dimension(embeddings_str)))
    with open(embeddings_str,'r', encoding='UTF-8') as f:
        for line in f:
            line_split = line.split()
            token = line_split[0]#.decode('utf-8')
            token = token.lower()
            idx = word2idx.get(token, 0)
            if idx > 0:
                embedding_vector = np.asarray(line_split[1:], dtype='float32')
                embedding_matrix[idx] = embedding_vector
        # code.interact(local=locals())
        return embedding_matrix

def load_data(A, X, Y, split_name, num_classes = 7):
    split_x = [[] for x in A[split_name][0]]
    split_y = [] 

    split_x[0] = [x.toarray()[0] for x in X[split_name]]
    for i in range(len(A[split_name][0]) - 1):
        for j in range(len(A[split_name])):
            split_x[i + 1].append(A[split_name][j][i].toarray())

    split_y = [to_categorical(y.toarray()[0], num_classes=7) for y in Y[split_name]]

    return split_x, split_y

def load_output(A, X, Y, split_name, num_classes = 7):
    split_y = [to_categorical(y.toarray()[0], num_classes=7) for y in Y[split_name]]

    return split_y

def batch_generator(A, X, Y, split_name, batch_size=16):
    num_sentences = len(X[split_name])
    batch_counter = 0

    split_x = [[] for x in A[split_name][0]]
    split_y = []

    while True:
        batch_start = batch_counter * batch_size
        batch_end = (batch_counter + 1) * batch_size
        split_x[0] = np.array([x.toarray()[0] for x in X[split_name][batch_start:batch_end]])
        for i in range(len(A[split_name][0]) - 1):
            for j in range(len(A[split_name][batch_start:batch_end])):
                split_x[i + 1].append(A[split_name][batch_start:batch_end][j][i].toarray())
            split_x[i + 1] = np.array(split_x[i + 1])
        split_y = np.array([to_categorical(y.toarray()[0], num_classes=7) for y in Y[split_name][batch_start:batch_end]])
        batch_counter = (batch_counter + 1) % (num_sentences // batch_size)
        yield split_x, split_y
        split_x = [[] for x in A[split_name][0]]
        split_y = []

base_path = 'H:/2019/ABSA/IMN-E2E-ABSA-master'
def create_vocab(domain, use_doc=True, maxlen=0, vocab_size=20000):
    file_list = [base_path + '%s_train.txt'%domain,
                 base_path + '%s_dev.txt'%domain,
                 base_path + '%s_test.txt'%domain]

    if use_doc:
        file_list.append('H:/2019/ABSA/IMN-E2E-ABSA-master/data_doc/electronics_large/text.txt')
        file_list.append('H:/2019/ABSA/IMN-E2E-ABSA-master/data_doc/yelp_large/text.txt')


    print ('Creating vocab ...')

    total_words, unique_words = 0, 0
    word_freqs = {}
    top = 0
    for f in file_list:
        top += 1
        fin = codecs.open(f, 'r', 'utf-8')
        for line in fin:
            words = line.strip().split()
            if maxlen > 0 and len(words) > maxlen:
                continue
            if len(words) < 2:
                continue
            if top < 4:
                w = words[0]
                if not is_number(w):
                    try:
                        word_freqs[w] += 1
                    except KeyError:
                        unique_words += 1
                        word_freqs[w] = 1
                    total_words += 1
            else:
                for w in words[:-1]:
            
            # print(w)
                    if not is_number(w):
                        try:
                            word_freqs[w] += 1
                        except KeyError:
                            unique_words += 1
                            word_freqs[w] = 1
                        total_words += 1
    fin.close()

    print ('  %i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>':0, '<unk>':1, '<num>':2}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print (' keep the top %i words' % vocab_size)

    return vocab