# author : bwlee@kbfg.com

import copy as cp

# it assumes that xs_ are integer categories
def onehotencode(xs_):
    min1, max1 = min(xs_), max(xs_)
    n_size = max1 - min1 + 1
    ret = []
    zeros0 = [0]*n_size
    for x1 in xs_:
        zeros = cp.copy(zeros0)
        zeros[x1-min1] = 1
        ret.append(zeros)
    return ret

def transpose(mat1):
    n_row = len(mat1)
    n_col = len(mat1[0])
    ret = [0]*n_col
    for j in range(n_col):
        ret[j] = [0]*n_row
        for i in range(n_row):
            ret[j][i] = mat1[i][j]
    return ret

if __name__ == '__main__':
    xs = [1,3,5,7]
    xs2 = [-1, -2, 2, 0]
    xsa = onehotencode(xs)
    xsb = onehotencode(xs2)
    print(xsa)
    print(xsb)

    import numpy as np
    xs3 = np.array(xs)
    xsc = onehotencode(xs3)
    print(xsc)

    import pandas as pd
    df1 = pd.DataFrame()
    df1['xs'] = xs
    df1['xs2'] = xs2
    xs4 = df1['xs']
    xsd = onehotencode(xs4)
    print(xsd)

    mat1 = [[1,3,5], [2,4,7]]
    mat2 = transpose(mat1)
    print(mat2)