from __future__ import print_function

import os, sys, math, operator
import numpy as np

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

class Word_dic:
    """
    For korean, we remove 'josa' 
    In addition, we will consider new words
    so don't care about 'UNK'
    MAX_count, MIN_count = 10000, 15
    """
    def __init__(self, texts = None, min_count = 10):
        self.min_count = min_count
        if texts:
            self.get_index(texts)
    
    def get_index(self, texts):
        self.word_count = {"<PAD>": 1, "<START>": 1, "<UNK>": 1}
        self.word2ix, self.ix2word = {"<PAD>": 0, "<START>": 1, "<UNK>": 2}, {0: "<PAD>", 1: "<START>", 2: "<UNK>"}
        for text1 in texts:
            for word1 in text1:
                self.word_count.setdefault(word1, 0)
                self.word_count[word1] += 1
        sorted_word_count = sorted(self.word_count.items(), key=operator.itemgetter(1), reverse=True)
        ix = 3 # keep first three indices
        for word1, _ in sorted_word_count:
            if self.word_count[word1] < self.min_count:
                break
            self.word2ix[word1] = ix
            self.ix2word[ix] = word1
            ix += 1
        self.words = list(self.word2ix.keys())
        self.n_words = len(self.words)
    
    def update(self, texts):
        """
        preserve original index and add oov words
        """
        new_words = self.find_oov_(texts)
        ix = self.n_words
        for word1, _ in new_words:
            self.word2ix[word1] = ix
            self.ix2word[ix] = word1
            ix += 1
        self.words = list(self.word2ix.keys())
        self.n_words = len(self.words)
        return np.array(new_words)[:,0]
        
    def find_oov_(self, texts):
        """
        return oov words and their counts
        """
        new_word_count = {}
        for text1 in texts:
            for word1 in text1:
                if word1 not in self.words:
                    new_word_count.setdefault(word1, 0)
                    new_word_count[word1] += 1
                self.word_count.setdefault(word1, 0)
                self.word_count[word1] += 1
        sorted_new_word_count = sorted(new_word_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_new_word_count    
    
if __name__ == '__main__':
    # we assume that texts are obtained from prreprocessing
    # texts = [text1], each text1 is a news article
    f2_s = '/home/bwlee2/work/projects/market_sensing/dict/cbow_update/texts.pk'
    word_index, sequences, texts = kload(f2_s)

    # get word_index 
    # ignore above variable 'word_index' which is obtained from other library
    # print(texts[0][:10])