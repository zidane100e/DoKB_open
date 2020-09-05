from collections import OrderedDict
from operator import itemgetter
import re, sys, os.path

from konlpy.tag import Okt
import numpy as np
import pickle

from kutils import list as list_util
from kutils import string as str_util

class Data():
    x_train, y_train = None, None
    x_test, y_test = None, None
    def __init__(self, label_type='binary'):
        self.type = label_type
        if self.type is 'binary':
            self.train_s = 'ratings_train.txt'
            self.test_s = 'ratings_test.txt'
            self.out_s = 'naver_movie.npz'
            self.out_id_s = 'naver_movie_id.pk'
        else:
            self.train_s = 'ratings_train_multi.txt'
            self.test_s = 'ratings_test_multi.txt'
            self.out_s = 'naver_movie_multi.npz'
            self.out_id_s = 'naver_movie_multi_id.pk'
        self.ix2word, self.word2ix, self.dic = None, None, None
        self.words = []

    def load_data(self):
        if not os.path.exists(self.out_s):
            self.preprocess()
        if Data.x_train is None:
            data0 = np.load(self.out_s)
            Data.x_train, Data.y_train = data0['train']
            Data.x_test, Data.y_test = data0['test']
        return (Data.x_train, Data.y_train), (Data.x_test, Data.y_test)

    def preprocess(self):
        x_train, y_train = self.preprocess_tokenize1(self.train_s)
        x_test, y_test = self.preprocess_tokenize1(self.test_s)
        self.preprocess_create_index2(x_train, x_test)

        Data.x_train, Data.y_train = self.preprocess_str2int(x_train, y_train)
        Data.x_test, Data.y_test = self.preprocess_str2int(x_test, y_test)

        np.savez(self.out_s, train=(Data.x_train, Data.y_train), test=(Data.x_test, Data.y_test))
        
    def preprocess_tokenize1(self, f1_s):
        xs, ys = self.read_txt(f1_s)
        xs = self.tokenize_set(xs)
        return xs, ys

    def preprocess_create_index2(self, *xs):
        if os.path.exists(self.out_id_s):
            with open(self.out_id_s, 'rb') as f_id:
                temp = pickle.load(f_id)
            self.word2ix = temp['word2ix']
            self.ix2word = temp['ix2word']
            self.dic = temp['dic']
        else:
            dic = list_util.count(xs[0] + xs[1])
            self.dic = OrderedDict(sorted(dic.items(), key=itemgetter(1), reverse=True))       
            self.ix2word, self.word2ix = self.indexing(self.dic)
            with open(self.out_id_s, 'wb') as f_id:
                pickle.dump({'word2ix': self.word2ix, 'ix2word': self.ix2word, 'dic': self.dic}, f_id)

    # transform string to integer form
    def preprocess_str2int(self, xs, ys):
        xs_i = [self.str2i(x, self.word2ix) for x in xs]
        return np.array(xs_i), np.array(ys)

    def read_txt(self, f1_s):
        map_strip = lambda x: x.strip()
        xs, ys = [], []
        i = 0
        with open(f1_s) as f1:
            f1.readline()
            for line in f1:
                elms = list(map(map_strip, line.strip().split('\t')))
                if len(elms) < 3:
                    continue
                xs.append(elms[1])
                ys.append(int(float(elms[2])))
                if i % 500 == 0:
                    print(i)
                #if i > 1000:
                #    break
                i += 1
        return xs, ys

    def tokenize(self, str1):
        analyzer = Okt()
        tokenized = analyzer.pos(str1, norm=True, stem=True)
        return ['/'.join(elm1) for elm1 in tokenized]

    def tokenize_set(self, arr_str1):
        ret = []
        for str1 in arr_str1:
            ret.append( self.tokenize(str1) )
            if len(ret) % 500 == 0:
                print(len(ret))
        return ret
        #return [self.tokenize(str1) for str1 in arr_str1]

    def indexing(self, ordereddic1):
        ii = 3  # save 0, 1, 2 for special key
        min_freq = 3
        ix2word_ = {0: '<pad>', 1: '<start>', 2: '<unk>'}
        word2ix_ = {'<pad>': 0, '<start>': 1, '<unk>': 2}
        for (word, freq) in ordereddic1.items():
            ix2word_[ii] = word
            word2ix_[word] = ii
            if freq < min_freq:
                break
            ii += 1
        self.ix2word = ix2word_
        self.word2ix = word2ix_
        return ix2word_, word2ix_

    def str2i(self, str1, word2ix_):
        filters = ['/Josa$', '/Number$', '/Determiner$', '/Suffix$', '/Punctuation$',
                   '/Exclamation', '/KoreanParticle']
        str2 = str_util.filter(str1, *filters)
        keys = word2ix_.keys()
        return [word2ix_[x] if x in keys else 2 for x in str2]

    def i2str(self, arr_is, ix2word_):
        return [ix2word_[i] for i in arr_is]

    def get_word_index(self):
        if self.ix2word is None:
            with open(self.out_id_s, 'rb') as f_id:
                temp = pickle.load(f_id)
            return {'word2ix': temp['word2ix'], 'ix2word': temp['ix2word'], 'dic': temp['dic']}
        else:
            return {'ix2word': self.ix2word, 'word2ix': self.word2ix, 'dic': self.dic}


if __name__ == '__main__':
    data1 = Data('multi')
    #data1 = Data()
    (x_train, y_train), (x_test, y_test) = data1.load_data()
    temp = data1.get_word_index()
    '''
    print(temp)
    dic, word2ix, ix2word = temp['dic'], temp['word2ix'], temp['ix2word']
    print(dic)
    print(word2ix)
    print(ix2word)
    print(type(ix2word))
    print(ix2word[0])
    print(ix2word[0], ix2word[5], ix2word[10])
    print(x_train)
    '''
