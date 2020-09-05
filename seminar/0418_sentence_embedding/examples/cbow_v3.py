from __future__ import print_function

import os, sys, math
import numpy as np

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Lambda, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dot, Reshape, Concatenate
from keras.models import Model
from keras.initializers import Constant
from keras.activations import sigmoid
from keras.callbacks import CSVLogger
from keras.models import load_model

from kutils.file import dump as kdump
from kutils.file import load as kload
from kutils.file import get_files

#sys.path.append('/home/bwlee2/research/embedding/word2vec/cbow_update')
sys.path.append(os.getcwd())
from word_index_v2 import Word_dic

EMBEDDING_DIM = 75
N_WINDOW = 3
N_NEGATIVE = 5
EPOCHS = 50
f2_s = '../data/text8'
with open(f2_s) as f2:
    texts0 = f2.read()
texts0 = texts0.split()
#texts = [texts0[:100000]]
texts = [texts0]
word_dic1 = Word_dic(texts, 15)
OPT = 'rmsprop'
#OPT = 'adam'
BATCH = 100
BATCH = 100
COEFF = 0.0001
#CONTEXT_WEIGHT = N_NEGATIVE
#CONTEXT_WEIGHT = N_WINDOW
CONTEXT_WEIGHT = 1.5
#CONTEXT_WEIGHT = 1


word2ix = word_dic1.word2ix

class Cbow:
    def __init__(self, n_window, n_negative, embed_dim, word_dic):
        self.n_window = n_window
        self.n_negative = n_negative
        self.embed_dim = embed_dim
        self.word_dic = word_dic
        self.n_words = word_dic.n_words
        self.min_count = self.word_dic.min_count
        self.network = None
        self.embed_name = 'embed'
        
    def get_negative_context(self, i_pos_context):
        # return negative sampling except context of i_pos
        neg_samples = []
        for ii in range(self.n_negative):
            while(True):
                sample = np.random.randint(low=3, high=self.n_words)
                if sample not in i_pos_context:
                    neg_samples.append(sample)
                    break
        return np.array(neg_samples)

    def get_negative(self, i_pos):
        # return negative sampling except i_pos
        # :examples
        # negative_sampling(20, 10, 2)
        samples = None
        while(True):
            samples = np.random.randint(low=3, high=self.n_words, size=self.n_negative)
            if i_pos not in samples:
                break
        return np.array(samples)

    def _prob_sub_sampling(self):
        prob_remove = {}
        coeff = COEFF
        for word1, count1 in self.word_dic.word_count.items():
            freq1 = count1/self.n_words
            prob_remove.setdefault(word1, 0)
            prob_remove[word1] = 1 - coeff/freq1 - math.sqrt(coeff/freq1)
        return prob_remove

    def _remove_min_count(self, texts_):
        for i, text1 in enumerate(texts_):
            # replace small counting words to '<UNK>'
            temp = []
            for word1 in text1:
                if self.word_dic.word_count[word1] > self.min_count:
                    temp.append(word1)
                else:
                    temp.append('<UNK>')
                    self.word_dic.word_count['<UNK>'] += 1
            texts_[i] = temp
        return texts_

    def _sub_sampling(self, texts_):
        prob_remove = self._prob_sub_sampling()
        for i, text1 in enumerate(texts_):
            # replace small counting words to 'unknown word'
            # texts_[i] = [word1 if self.word_dic.word_count[word1] > self.min_count else '<UNK>' for word1 in text1]
            temp = []
            for word1 in text1:
                if self.word_dic.word_count[word1] > self.min_count:
                    temp.append(word1)
                else:
                    temp.append('<UNK>')
                    self.word_dic.word_count['<UNK>'] += 1
            # subsampling
            text2 = []
            for word1 in temp:
                prob1 = prob_remove[word1]
                rand = np.random.random()
                #print('rand', rand, 'prob', prob1)
                if rand < prob1:
                    continue
                else:
                    text2.append(word1)
            texts_[i] = text2
        return texts_

    # pad for front anc back side
    def get_context(self, i, text1_):
        n_window = self.n_window
        ret = None
        if i - n_window < 0:
            ret = ["<PAD>"] * (n_window - i) + text1_[:i]
        else:
            ret = text1_[i - n_window:i]
        if i + n_window + 1 > len(text1_):
            ret += text1_[i+1:] + ["<PAD>"] * (i + n_window + 1 - len(text1_))
        else :
            ret += text1_[i+1:i + n_window + 1]
        return [ self.word_dic.word2ix[word1] for word1 in ret ]    

    # make input data
    # data types are input : [[x1]], negative : [[x1n1, x1n2, x1n3, ...], ], context : [[x1c1, x1c2, x1c3], ]
    def get_train_data(self, texts):
        texts = self._remove_min_count(texts)
        prob_remove = self._prob_sub_sampling()
        self.data_x = []
        self.data_context = []
        self.data_negative = []
        for i, text1 in enumerate(texts):
            rands = np.random.random(len(text1))
            for j, word1 in enumerate(text1):
                prob1 = prob_remove[word1]
                #rand = np.random.random()
                #print('rand', rand, 'prob', prob1)
                if rands[j] < prob1:
                    continue
                else:
                    i_pos = self.word_dic.word2ix[word1]
                    self.data_x.append([i_pos])
                    self.data_context.append(self.get_context(j, text1))
                    #context_temp = list(self.data_context[-1]) + [i_pos]
                    #self.data_negative.append(self.get_negative_context(context_temp))
                    self.data_negative.append(self.get_negative(i_pos))
        self.data_x = np.array(self.data_x)
        self.data_context = np.array(self.data_context)
        self.data_negative = np.array(self.data_negative)
        self.target_data = np.ones( self.data_x.size )
        self.target_negative = np.zeros((self.data_x.size, self.n_negative))
        return [self.data_x, self.data_context, self.data_negative], [self.target_data, self.target_negative]
        
    def get_network(self, mat1_ = None, flag_trainable = True):
        if self.network is not None:
            return self.network
        
        data_in = Input(shape=(1,))
        context_in = Input(shape=(self.n_window*2, ))
        negative_in = Input(shape=(self.n_negative, ))
        
        if mat1_ is not None:
            embedded = Embedding(input_dim=self.n_words, output_dim=self.embed_dim, name=self.embed_name, 
                                 trainable = flag_trainable, embeddings_initializer=Constant(mat1_))
        else:
            embedded = Embedding(input_dim=self.n_words, output_dim=self.embed_dim, name=self.embed_name, 
                                 trainable = flag_trainable)
            
        word_embedded = embedded(data_in)
        context_embedded = embedded(context_in)
        negative_embedded = embedded(negative_in)

        cbow = Lambda(lambda x: K.mean(x, axis=1), output_shape=(self.embed_dim,))(context_embedded)

        hidden1 = Dot(-1, normalize=True)([word_embedded, cbow])
        hidden2 = Dot(-1, normalize=True)([negative_embedded, cbow])

        #out1 = Dense(1, activation='sigmoid', name='out1')(hidden1)
        out1 = Activation('sigmoid', name='out_context')(hidden1)
        # negative sampling has n_negative dimension
        # so use lambda instead of Dense
        
        #out2 = []
        #for kk in range(self.n_negative):
        #    lam1 = Lambda(lambda x: x[kk])(hidden2)
        #    out2.append( Dense(1, activation='sigmoid')(lam1))
        #out2 = Concatenate()(out2)
        
        #out2a = Reshape((1,self.n_negative), input_shape=(self.n_negative,))(hidden2)
        #out2 = Dense(1, activation='sigmoid')(out2a)
        out2 = Activation('sigmoid', name='out_neg')(hidden2)
        #out2 = Lambda(lambda x: sigmoid(x))(hidden2)
        
        self.network = Model(inputs=[data_in, context_in, negative_in], outputs=[out1, out2])
        return self.network
    
    def set_embed(self, mat1_):
        """
        this is for update text
        preserve embed_dim and only change n_words
        """
        self.network = None
        self.n_words = mat1_.shape[0]
        self.get_network(mat1_)
    
    def get_embed(self):
        self.get_network()
        return self.network.get_layer(self.embed_name).get_weights()[0]
    
cbow1 = Cbow(n_window=N_WINDOW, n_negative=N_NEGATIVE, embed_dim=EMBEDDING_DIM, word_dic=word_dic1)

input, target = cbow1.get_train_data(texts)

model1 = cbow1.get_network()
model1.compile(optimizer='rmsprop', loss='binary_crossentropy')
print(model1.summary())

import nmslib
class SimilarCallback:
    def __init__(self, embed1):
        #self.words_test = embed1.word_dic.words[50:70]
        #self.words_test = np.array(embed1.word_dic.words)[np.random.randint(50, min(1000, cbow1.word_dic.n_words), 20)]
        self.words_test = np.array(['return', 'move', 'ball', 'records', 'including', 'largest', 'range', 'night', 'types', 'south', 'august', 'upon', 'peace', 'version', 'back', 'earlier', 'objects', 'service', 'nuclear'])
        self.index = self.reset(embed1)
        self.embed = embed1

    def reset(self, embed1):
        self.mat = embed1.get_embed()
        # initialize a new index, using a HNSW index on Cosine Similarity
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.addDataPointBatch(self.mat)
        self.index.createIndex({'post': 2}, print_progress=True)
        return self.index

    def run_sim(self):
        # query for the nearest neighbours of the first datapoint
        res = ''
        for word1 in self.words_test:
            ids1, distances1 = self.index.knnQuery(self.mat[self.embed.word_dic.word2ix[word1]], k=10)
            words2 = []
            for id1 in ids1:
                words2.append(self.embed.word_dic.ix2word[id1])
            res += word1
            res += " : "
            res += ', '.join(words2) + '\n'
        return res
            
model0 = cbow1.get_network()
#model0 = load_model('cbow_dim75_rmsprop_batch_100_COEF_0.001_cont_weigt_1.h5')
# multi GPU
from keras.utils import multi_gpu_model
model1 = multi_gpu_model(model0, gpus=2)
#model1 = model0
model1.compile(optimizer=OPT, loss='binary_crossentropy', loss_weights = {'out_neg': 1, 'out_context': CONTEXT_WEIGHT} )

f2_s = 'cbow_dim' + str(EMBEDDING_DIM) + '_' + OPT + '_batch_' + str(BATCH) + '_COEF_' + str(COEFF) + '_cont_weigt_' + str(CONTEXT_WEIGHT) + '.log'
csv_logger = CSVLogger(f2_s)


step = 3
sim1 = SimilarCallback(cbow1)
for ii in range(0, EPOCHS, step):
    #score = model1.fit(x=[cbow1.data_x, cbow1.data_context, cbow1.data_negative], y=[cbow1.target_data, cbow1.target_negative], batch_size=100, epochs= step, callbacks=[csv_logger])
    score = model1.fit(x=[cbow1.data_x, cbow1.data_context, cbow1.data_negative], y=[cbow1.target_data, cbow1.target_negative], batch_size=BATCH, epochs= step)
    sim1.reset(cbow1)
    ret = sim1.run_sim()
    with open(f2_s, 'a') as f2:
        f2.write(str(score.history) + '\n')
        f2.write(ret)
        f2.write('\n')
    
    

f2_s_out = f2_s.replace('.log', '.h5')
model1.save(f2_s_out)
