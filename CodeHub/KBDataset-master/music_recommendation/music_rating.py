# users : 200000 
# musics : 127771
import os

import numpy as np

class Data():
    x_train, y_train = None, None
    x_test, y_test = None, None
    Test_cut = 160000
    F_in_s = 'songsDataset.csv'
    F_out_s = 'song_ratings.npz'

    def load_data(self):
        if os.path.exists(Data.F_out_s):
            data0 = np.load(Data.F_out_s)
            Data.x_train, Data.y_train = data0['xs_train'], data0['ys_train']
            Data.x_test, Data.y_test = data0['xs_test'], data0['ys_test']
        elif Data.x_train is None:
            self.preprocess()
        return (Data.x_train, Data.y_train), (Data.x_test, Data.y_test)

    def preprocess(self):
        # i_test is an index for test data are started
        # original data are arranged by user
        # so cut at a user which exceeds the number of Data.Test_cut
        users, songs, ys, i_test = self.read_txt(Data.F_in_s)
        user_train, user_test = users[:i_test], users[i_test:]
        song_train, song_test = songs[:i_test], songs[i_test:]
        y_train, y_test = ys[:i_test], ys[i_test:]
        
        # shuffle train_data
        ids = list(range(user_train.shape[0]))
        np.random.shuffle(ids)
        user_train = user_train[ids]
        song_train = song_train[ids]
        y_train = y_train[ids]

        Data.x_train = [user_train, song_train]
        Data.x_test = [user_test, song_test]
        Data.y_train = y_train
        Data.y_test = y_test

        np.savez(Data.F_out_s, xs_train=Data.x_train, ys_train=Data.y_train, xs_test=Data.x_test, ys_test=Data.y_test)

    def read_txt(self, f1_s):
        ys, users, songs = [], [], []
        i_test = 0
        with open(f1_s) as f1:
            f1.readline()
            for line in f1:
                elms = line.split(',')
                users.append([int(elms[0])])
                songs.append([int(elms[1])])            
                ys.append(float(elms[2]))
                # get row index to separate train and test data
                if int(elms[0]) < Data.Test_cut:
                    i_test += 1
        return np.array(users), np.array(songs), np.array(ys), i_test

if __name__ == '__main__':
    data1 = Data()
    (x_train, y_train), (x_test, y_test) = data1.load_data()
    print(x_train[0])
    print(x_train[1])
    print(y_train)
    print(x_test[1])
    print(y_test)