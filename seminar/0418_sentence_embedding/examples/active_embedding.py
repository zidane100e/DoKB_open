from __future__ import print_function

import os, sys, math, operator
import numpy as np, copy as cp

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dot
from keras.models import Model
from keras.initializers import Constant
from keras.activations import sigmoid

from kutils.file import dump as kdump
from kutils.file import load as kload
from kutils.file import get_files

sys.path.append(os.getcwd())
from word_index import Word_dic
from cbow import Cbow

class KEmbedding(Cbow):
    def __init__(self, n_window, n_negative, embed_dim):
        self.n_window, self.n_negative, self.embed_dim = n_window, n_negative, embed_dim
        self.word_dic = None

    def _concat_tuple(texts):
        """
        if tuple of texts(array of array of string) is given,
        it concatenates all tuple in texts format
        """
        temp = []
        for text1 in texts:
            temp += text1
        texts = temp
        return texts
        
    def load_text(self, *texts):
        """
        initialize dictionary from texts
        """
        texts = KEmbedding._concat_tuple(texts)
        self.word_dic = Word_dic(texts)
        Cbow.__init__(self, self.n_window, self.n_negative, self.embed_dim, self.word_dic)
        self.texts = texts
        
    def add_text(self, *texts):
        """
        preserve self.word_dic and add words after that
        """            
        if self.word_dic is None:
            self.load_text(*texts)
            self.get_network()
        else:
            texts = KEmbedding._concat_tuple(texts)
            
            mat_old = self.get_embed()
            n_words_old = self.n_words
            word_dic_old = cp.copy(self.word_dic)
            self.new_words = self.word_dic.update(texts)
            
            Cbow.__init__(self, self.n_window, self.n_negative, self.embed_dim, self.word_dic)
            n_words_new = self.n_words
            word_dic_new = self.word_dic

            new_word_context_dic = { word1: self._contexts_of(word1, texts) for word1 in self.new_words }
            if n_words_old == n_words_new:
                mat_new = mat_old
            else:
                mat_new = []
                for ix1 in range(n_words_old, n_words_new):
                    word2 = self.word_dic.ix2word[ix1]
                    embed2 = self._get_avg_embed(word2, mat_old, word_dic_old, new_word_context_dic)
                    mat_new.append(embed2)
                mat_new = np.array(mat_new)
                mat_new = np.concatenate((mat_old, mat_new))        
            
            self.set_embed(mat_new)
            self.texts = texts
    
    def get_train_data(self):
        return super().get_train_data(self.texts)
    
    def clone(self):
        return cp.copy(self)
    
    def _contexts_of(self, word1a, texts2, n_window=3):
        """
        get top frequent context of word1a from texts2
        """
        context = {}
        def add_count_dic(count_dic1, key_arr1):
            for key1 in key_arr1:
                count_dic1.setdefault(key1, 0)
                count_dic1[key1] += 1
        def get_top_words(count_dic1):
            sorted_count_dic1 = sorted(count_dic1.items(), key=operator.itemgetter(1), reverse=True)
            n_words = len(sorted_count_dic1)
            n_words2 = max(10, int(n_words/10)) # consider 10% of words in calculation
            return { word1: count1 for word1, count1 in sorted_count_dic1[:n_words2] }

        for text1 in texts2:
            n_text1 = len(text1)
            for i, word1 in enumerate(text1):
                if word1a == word1:
                    low_limit = max([i-n_window, 0])
                    high_limit = min([i+n_window+1, n_text1])
                    arr_temp = text1[low_limit:i]
                    arr_temp += text1[i+1:high_limit]
                    add_count_dic(context, arr_temp)
        return get_top_words(context)

    def _get_avg_embed(self, word0, embed_mat1, word_dic1, word_context_dic):
        """
        get embedding of oov word0 by averaging its already seen context
        """
        context_count1 = word_context_dic[word0]        
        words = word_dic1.words
        total_count_sum = 0
        embed = np.zeros(self.embed_dim)
        for word1, count1 in context_count1.items():
            if word1 in words:
                ix1 = word_dic1.word2ix[word1]
                embed += embed_mat1[ix1] * count1
                total_count_sum += count1
        if total_count_sum < 1:
            return np.zeros(self.embed_dim)
        return embed/total_count_sum
        

if __name__ == '__main__':
    f2_s = '/home/bwlee2/work/projects/market_sensing/dict/cbow_update/texts.pk'
    _, _, texts = kload(f2_s)

    EMBEDDING_DIM = 100
    N_WINDOW = 3
    N_NEGATIVE = 5

    texts1 = texts[:15000]

    embed1 = KEmbedding(N_WINDOW, N_NEGATIVE, EMBEDDING_DIM)
    embed1.load_text(texts1)
    
    input1, target1 = embed1.get_train_data()
    model1 = embed1.get_network()
    model1.compile(optimizer='rmsprop', loss='binary_crossentropy')
    score1 = model1.fit(x=input1, y=target1, batch_size=100, epochs=20)
    
    mat1 = embed1.get_embed()
    
    texts2 = texts[15000:]

    embed2 = embed1.clone()
    embed2.add_text(texts2)
    
    input2, target2 = embed2.get_train_data()
    model2 = embed2.get_network()
    model2.compile(optimizer='rmsprop', loss='binary_crossentropy')
    score2 = model2.fit(x=input2, y=target2, batch_size=100, epochs=20)
    
    import nmslib

    mat1 = embed1.get_embed()
    mat2 = embed2.get_embed()

    # initialize a new index, using a HNSW index on Cosine Similarity
    index1 = nmslib.init(method='hnsw', space='cosinesimil')
    index1.addDataPointBatch(mat1)
    index1.createIndex({'post': 2}, print_progress=True)

    index2 = nmslib.init(method='hnsw', space='cosinesimil')
    index2.addDataPointBatch(mat2)
    index2.createIndex({'post': 2}, print_progress=True)

    # query for the nearest neighbours of the first datapoint
    word1 = 'directly'
    word2 = 'trouble'

    ids1, distances1 = index1.knnQuery(mat1[embed1.word_dic.word2ix[word1]], k=10)
    print('\ndirectly ----------------')
    for id1 in ids1:
        print(embed1.word_dic.ix2word[id1])

    ids2a, distances2a = index2.knnQuery(mat2[embed2.word_dic.word2ix[word1]], k=10)
    ids2b, distances2b = index2.knnQuery(mat2[embed2.word_dic.word2ix[word2]], k=10)
    print('\ndirectly ----------------')
    for id1 in ids2a:
        print(embed2.word_dic.ix2word[id1])
        
    print('\ntrouble ----------------')
    for id1 in ids2b:
        print(embed2.word_dic.ix2word[id1])
